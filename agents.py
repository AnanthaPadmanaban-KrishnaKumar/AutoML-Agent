import os
import uuid
import json
import logging
import textwrap
from typing import Dict, Any, Optional, List, Tuple

from openai import OpenAI
import pinecone

try:
    from autogen import AssistantAgent, UserProxyAgent, Agent, ConversableAgent
    from autogen.coding import LocalCommandLineCodeExecutor
except ImportError:
    print("Please install required libraries")
    exit(1)

# Import configurations and initialized services
from config import (
    PLANNER_LLM_CONFIG, PLANNER_LLM_NAME,
    CODE_GEN_LLM_CONFIG, CODE_GEN_LLM_NAME,
    EMBEDDING_MODEL, PINECONE_TOP_K,
    CODE_EXECUTION_DIR, CODE_EXEC_TIMEOUT, logger
)
from services import pinecone_index, openai_client

# --- Custom Memory Agent Definition ---
class PineconeMemoryAgent(ConversableAgent):
    """
    Agent for storing/retrieving text data using Pinecone vector embeddings.
    Handles JSON requests for STORE and RETRIEVE actions.
    """
    def __init__(self,
                 name: str = "memory_manager",
                 pinecone_index_instance: pinecone.Index = pinecone_index,
                 embedding_client_instance: OpenAI = openai_client,
                 embedding_model: str = EMBEDDING_MODEL,
                 top_k: int = PINECONE_TOP_K) -> None:
        super().__init__(name=name)
        if not pinecone_index_instance or not embedding_client_instance:
             raise ValueError("PineconeMemoryAgent requires initialized pinecone_index and openai_client.")
        self.index = pinecone_index_instance
        self.embedding_client = embedding_client_instance
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.register_reply(Agent, self._handle_request_message, position=1)
        logger.info(f"PineconeMemoryAgent '{self.name}' initialized (Embeddings: {self.embedding_model}, TopK: {self.top_k}).")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of non-empty texts."""
        processed_texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not processed_texts:
            logger.warning("Embedding requested for empty or invalid text list.")
            return []
        try:
            response = self.embedding_client.embeddings.create(input=processed_texts, model=self.embedding_model)
            embedding_map = {text: data.embedding for text, data in zip(processed_texts, response.data)}
            return [embedding_map[text] for text in processed_texts]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise 

    def _handle_store_request(self, payload: Dict[str, Any], sender: Agent) -> None:
        """Handles internal logic for storing text data."""
        texts_to_store = payload.get("texts", [])
        valid_texts = [t.strip() for t in texts_to_store if isinstance(t, str) and t.strip()]

        if not valid_texts:
            logger.warning(f"STORE request from {sender.name} contained no valid text.")
            self.send(message="ACTION_FAILED: No valid, non-empty text provided to store.", receiver=sender)
            return

        logger.debug(f"Handling STORE request for {len(valid_texts)} texts from {sender.name}.")
        try:
            embeddings = self._embed(valid_texts)
            if len(embeddings) != len(valid_texts):
                raise ValueError("Mismatch between number of valid texts and generated embeddings.")

            vectors = [(str(uuid.uuid4()), vec, {"text": txt}) for txt, vec in zip(valid_texts, embeddings)]
            if not vectors:
                 raise ValueError("No valid vectors generated after embedding.")

            upsert_response = self.index.upsert(vectors=vectors)
            logger.info(f"Upserted {upsert_response.upserted_count}/{len(vectors)} items into Pinecone for {sender.name}.")
            self.send(message=f"ACTION_SUCCESS: Stored {upsert_response.upserted_count} items in memory.", receiver=sender)

        except Exception as e:
            logger.error(f"Pinecone STORE operation failed for {sender.name}: {e}", exc_info=True)
            self.send(message=f"ACTION_FAILED: Failed to store data in Pinecone memory. Details: {e}", receiver=sender)

    def _handle_retrieve_request(self, payload: Dict[str, Any], sender: Agent) -> None:
        """Handles internal logic for retrieving text data."""
        query = payload.get("query", "").strip()
        if not query:
             logger.error(f"RETRIEVE request from {sender.name} had empty query.")
             self.send(message="ACTION_FAILED: RETRIEVE requires a non-empty 'query'.", receiver=sender)
             return

        logger.debug(f"Handling RETRIEVE request for query '{query}' from {sender.name}.")
        try:
            embedding_result = self._embed([query])
            if not embedding_result:
                raise ValueError(f"Failed to generate embedding for query: '{query}'")
            embedding = embedding_result[0]

            result = self.index.query(vector=embedding, top_k=self.top_k, include_metadata=True)
            matches = [(m.metadata.get("text", "N/A"), m.score) for m in getattr(result, 'matches', []) if m.metadata]

            if not matches:
                logger.info(f"No results found in Pinecone for query: '{query}' from {sender.name}")
                self.send(message="ACTION_SUCCESS: No relevant information found in memory for your query.", receiver=sender)
            else:
                formatted_results = "\n".join([f"- (Score: {score:.4f}) {txt}" for txt, score in matches])
                logger.info(f"Retrieved {len(matches)} results for query: '{query}' from {sender.name}")
                self.send(message=f"ACTION_SUCCESS: Retrieved Information from Memory:\n{formatted_results}", receiver=sender)
        except Exception as e:
            logger.error(f"Pinecone RETRIEVE operation failed for query '{query}' from {sender.name}: {e}", exc_info=True)
            self.send(message=f"ACTION_FAILED: Failed to retrieve data from Pinecone memory. Details: {e}", receiver=sender)

    def _handle_request_message(self, messages: Optional[List[Dict]] = None, sender: Optional[Agent] = None, config: Optional[Any] = None) -> Tuple[bool, Optional[Any]]:
        """Parses incoming JSON messages and dispatches to specific handlers."""
        # Check if message is intended for this agent and is valid
        if not sender or not messages or self.name != messages[-1].get("recipient", self.name):
            return False, None 
        last_message = messages[-1]
        content = last_message.get("content", "")
        action, payload, error_msg = None, None, None

        logger.debug(f"MemoryAgent received message from {sender.name}: {content[:200]}...")
        # --- Parse and Validate JSON Request ---
        try:
            data = json.loads(content)
            if not isinstance(data, dict):
                error_msg = "ACTION_FAILED: Request must be a JSON object."
            else:
                action = data.get("action")
                payload = data.get("payload")
                if not action or action not in ["STORE", "RETRIEVE"]:
                     error_msg = f"ACTION_FAILED: Invalid or missing 'action'. Must be 'STORE' or 'RETRIEVE'."
                elif not isinstance(payload, dict):
                     error_msg = f"ACTION_FAILED: Invalid 'payload'. Must be a JSON object."

        except json.JSONDecodeError:
            error_msg = "ACTION_FAILED: Invalid JSON format."
        except Exception as e:
            error_msg = f"ACTION_FAILED: Error parsing request: {e}"

        # --- Action Dispatching & Payload Validation ---
        if not error_msg:
            if action == "STORE":
                texts = payload.get("texts")
                if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                    error_msg = "ACTION_FAILED: STORE 'payload.texts' must be a list of strings."
                elif not any(t.strip() for t in texts):
                    error_msg = "ACTION_FAILED: STORE 'payload.texts' must contain at least one non-empty string."
                else:
                    self._handle_store_request(payload, sender)
                    return True, None 

            elif action == "RETRIEVE":
                query = payload.get("query")
                if not isinstance(query, str) or not query.strip():
                    error_msg = "ACTION_FAILED: RETRIEVE 'payload.query' must be a non-empty string."
                else:
                    payload["query"] = query.strip()
                    self._handle_retrieve_request(payload, sender)
                    return True, None 

        # --- Handle Errors ---
        if error_msg:
            logger.error(f"MemoryAgent Error: {error_msg} (From: {sender.name}, Content: {content[:200]}...)")
            self.send(message=error_msg, receiver=sender)
            return True, None

        # Fallback if something unexpected happens
        logger.warning(f"Memory agent handler reached unexpected state for action '{action}'.")
        return False, None 

# --- Agent Creation Functions ---

# def create_executor_agent(name: str = "executor") -> UserProxyAgent:
#     """Creates the code execution agent."""
#     os.makedirs(CODE_EXECUTION_DIR, exist_ok=True)
#     executor = UserProxyAgent(
#         name=name,
#         human_input_mode="NEVER",
#         llm_config=False,
#         code_execution_config={
#             "executor": LocalCommandLineCodeExecutor(timeout=CODE_EXEC_TIMEOUT, work_dir=CODE_EXECUTION_DIR),
#             "use_docker": False # Set to True if Docker is preferred and installed
#         },
#         system_message="You are a code executor. You receive Python code, execute it, and return the result. Prefix the output with EXECUTION_SUCCESS: or EXECUTION_FAILED:. Provide the full stdout and stderr in case of failure.",
#     )
#     logger.info(f"Executor agent '{name}' initialized (Work Dir: ./{CODE_EXECUTION_DIR}).")
#     return executor

def create_executor_agent(name: str = "executor") -> UserProxyAgent:
    """Creates the code execution agent."""
    os.makedirs(CODE_EXECUTION_DIR, exist_ok=True)
    executor = UserProxyAgent(
        name=name,
        human_input_mode="NEVER",
        llm_config=False,
        code_execution_config={
            # Provide the executor instance directly
            "executor": LocalCommandLineCodeExecutor(timeout=CODE_EXEC_TIMEOUT, work_dir=CODE_EXECUTION_DIR),
            # Remove the "use_docker" key from this level
        },
        system_message="You are a code executor. You receive Python code, execute it, and return the result. Prefix the output with EXECUTION_SUCCESS: or EXECUTION_FAILED:. Provide the full stdout and stderr in case of failure.",
    )
    logger.info(f"Executor agent '{name}' initialized (Work Dir: ./{CODE_EXECUTION_DIR}).")
    return executor
    
def create_code_generator_agent(name: str = "code_generator") -> AssistantAgent:
    """Creates the code generation specialist agent."""
    code_generator = AssistantAgent(
        name=name,
        llm_config=CODE_GEN_LLM_CONFIG, 
        system_message=textwrap.dedent(
            f"""
            You are a specialized Python code generation assistant using {CODE_GEN_LLM_NAME}.
            Your goal is to write correct, executable Python code based on the user's request.
            **Output Format:** Respond ONLY with the Python code block. Start the block with ```python and end it with ```. Do not include any other text, explanations, or introductory phrases before or after the code block.
            **Error Handling:** If asked to fix code, analyze the provided original code and error. Output ONLY the corrected Python code block (```python ... ```).
            """
        ),
    )
    logger.info(f"Code Generator agent '{name}' initialized with {CODE_GEN_LLM_NAME}.")
    return code_generator

def create_planner_agent(name: str = "planner", memory_agent_name: str = "memory_manager", code_gen_name: str = "code_generator", exec_name: str = "executor") -> AssistantAgent:
    """Creates the central planner agent."""
    planner = AssistantAgent(
        name=name,
        llm_config=PLANNER_LLM_CONFIG,
        system_message=textwrap.dedent(
            f"""
            You are a central planner agent using {PLANNER_LLM_NAME}. Orchestrate tasks using tools to interact with specialist agents: {memory_agent_name}, {code_gen_name}, {exec_name}.

            **Available Tools (Function Calls):**
            1. `store_texts_in_memory(texts: List[str])`: Request {memory_agent_name} to store text.
            2. `retrieve_texts_from_memory(query: str)`: Request {memory_agent_name} to search text.
            3. `generate_python_code(task_description: str)`: Request {code_gen_name} to write/fix Python code.
            4. `execute_python_code(code: str)`: Request {exec_name} to execute Python code.

            **Workflow:**
            1. Analyze request.
            2. Plan & Call ONE tool (validate args first). Tool returns "STATUS: Request sent...".
            3. **WAIT** for the result message from the specialist agent (`ACTION_SUCCESS/FAILED`, `EXECUTION_SUCCESS/FAILED`, or code block).
            4. Analyze result:
               - `ACTION_SUCCESS`: Use results/confirmation.
               - `ACTION_FAILED`: Report error or retry.
               - Code block: Call `execute_python_code`.
               - `EXECUTION_SUCCESS`: Analyze output.
               - `EXECUTION_FAILED`: Call `generate_python_code` with goal, faulty code, and FULL error. Retry 1-2 times max. Report final error if needed.
            5. Respond to user only when finished (no tool call in final response).

            **Rules:** Always wait for agent response after sending a request. Include full error details for fixes. Validate args. Be methodical.
            """
        ),
    )
    logger.info(f"Planner agent '{name}' initialized with {PLANNER_LLM_NAME}.")
    return planner

def create_user_proxy_agent(name: str = "user_proxy") -> UserProxyAgent:
    """Creates the agent responsible for interacting with the human user."""
    user_proxy = UserProxyAgent(
        name=name,
        human_input_mode="ALWAYS", # Ask human each turn
        code_execution_config=False,
        llm_config=False,
        default_auto_reply="Okay, forwarding your request to the planner...",
        system_message="You are the user interface. Pass the user's request to the planner. Report the planner's final response back to the user. Terminate the chat if the user asks to exit (e.g., 'exit', 'quit', 'stop').",
    )
    logger.info(f"User Proxy agent '{name}' initialized.")
    return user_proxy

# --- Termination Function ---
def is_termination_msg(content: Any) -> bool:
    """Checks if the message content signals termination."""
    if isinstance(content, dict):
        content = content.get("content", "")
    if isinstance(content, str):
        return content.strip().lower() in ["exit", "quit", "terminate", "stop", "goodbye"]
    return False