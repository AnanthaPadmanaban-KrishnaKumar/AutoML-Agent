import json
import logging
from typing import List, Dict, Any

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from autogen import Agent

logger = logging.getLogger(__name__)

def store_texts_in_memory(texts: List[str], planner_agent: 'Agent', memory_agent: 'Agent') -> str:
    """
    Tool function called by the Planner LLM.
    Sends a structured JSON request to the memory_manager agent to store texts.
    Requires planner_agent and memory_agent instances to be passed.
    """
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        logger.error("Tool 'store_texts_in_memory': Invalid 'texts' argument.")
        return "ACTION_FAILED: Input 'texts' must be a list of strings."

    valid_texts = [t.strip() for t in texts if t.strip()]
    if not valid_texts:
        logger.warning("Tool 'store_texts_in_memory': No valid non-empty texts provided.")
        return "ACTION_FAILED: No valid non-empty texts provided to store."

    logger.info(f"Tool: Requesting {memory_agent.name} to store {len(valid_texts)} texts.")
    message_payload = json.dumps({
        "action": "STORE",
        "payload": {"texts": valid_texts}
    })
    # Send message FROM planner TO memory_agent
    planner_agent.send(message=message_payload, recipient=memory_agent)
    return f"STATUS: Request sent to {memory_agent.name} to store texts. Waiting for ACTION_SUCCESS/ACTION_FAILED confirmation from {memory_agent.name}."

def retrieve_texts_from_memory(query: str, planner_agent: 'Agent', memory_agent: 'Agent') -> str:
    """
    Tool function called by the Planner LLM.
    Sends a structured JSON request to the memory_manager agent to retrieve texts.
    Requires planner_agent and memory_agent instances to be passed.
    """
    if not isinstance(query, str) or not query.strip():
        logger.error("Tool 'retrieve_texts_from_memory': Invalid 'query' argument.")
        return "ACTION_FAILED: Input 'query' must be a non-empty string."
    cleaned_query = query.strip()

    logger.info(f"Tool: Requesting {memory_agent.name} retrieval for query: '{cleaned_query}'")
    message_payload = json.dumps({
        "action": "RETRIEVE",
        "payload": {"query": cleaned_query}
    })
     # Send message FROM planner TO memory_agent
    planner_agent.send(message=message_payload, recipient=memory_agent)
    return f"STATUS: Request sent to {memory_agent.name} for query '{cleaned_query}'. Waiting for ACTION_SUCCESS/ACTION_FAILED results from {memory_agent.name}."

def generate_python_code(task_description: str, planner_agent: 'Agent', code_generator_agent: 'Agent') -> str:
    """
    Tool function called by the Planner LLM.
    Sends a task description to the code_generator agent.
    Requires planner_agent and code_generator_agent instances to be passed.
    """
    if not isinstance(task_description, str) or not task_description.strip():
        logger.error("Tool 'generate_python_code': Invalid 'task_description'.")
        return "ACTION_FAILED: Input 'task_description' must be a non-empty string."
    cleaned_task = task_description.strip()

    logger.info(f"Tool: Requesting {code_generator_agent.name} for task: '{cleaned_task[:100]}...'")
     # Send message FROM planner TO code_generator
    planner_agent.send(message=cleaned_task, recipient=code_generator_agent)
    return f"STATUS: Request sent to {code_generator_agent.name}. Waiting for the Python code block response from {code_generator_agent.name}."

def execute_python_code(code: str, planner_agent: 'Agent', executor_agent: 'Agent') -> str:
    """
    Tool function called by the Planner LLM.
    Sends Python code (as a string) to the executor agent for execution.
    Requires planner_agent and executor_agent instances to be passed.
    """
    if not isinstance(code, str):
        logger.error("Tool 'execute_python_code': Invalid 'code' argument.")
        return "ACTION_FAILED: Input 'code' must be a string."

    cleaned_code = code.strip()
    if cleaned_code.startswith("```python"):
        cleaned_code = cleaned_code[len("```python"):].strip()
    if cleaned_code.endswith("```"):
        cleaned_code = cleaned_code[:-len("```")].strip()
    if cleaned_code.startswith("```") and cleaned_code.endswith("```"):
         if len(cleaned_code):
              cleaned_code = cleaned_code[3:-3].strip()

    if not cleaned_code:
        logger.error("Tool 'execute_python_code': No valid code after cleaning.")
        return "ACTION_FAILED: No valid Python code provided."

    logger.info(f"Tool: Requesting {executor_agent.name} to run code:\n{cleaned_code[:500]}...")
     # Send message FROM planner TO executor
    planner_agent.send(message=cleaned_code, recipient=executor_agent)
    return f"STATUS: Code sent to {executor_agent.name}. Waiting for EXECUTION_SUCCESS/EXECUTION_FAILED results from {executor_agent.name}."