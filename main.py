import logging
import sys
import functools
from datetime import datetime

import config
import tools
from services import pinecone_index, openai_client

from agents import (
    PineconeMemoryAgent,
    create_executor_agent,
    create_code_generator_agent,
    create_planner_agent,
    create_user_proxy_agent,
    is_termination_msg
)

logger = logging.getLogger(__name__)

def register_termination_handlers(agents: list):
    """Registers the termination message handler for relevant agents."""
    from autogen import Agent
    for agent in agents:
         if hasattr(agent, 'register_reply'):
              logger.debug(f"Registering termination handler for {agent.name}")
              agent.register_reply(
                   Agent,
                   reply_func=lambda messages, sender, config: (True, "TERMINATE") if is_termination_msg(messages[-1].get("content")) else (False, None),
                   config={},
                   trigger=lambda msg: is_termination_msg(msg.get("content") if isinstance(msg, dict) else msg) # Check trigger based on content
              )


def run_chat():
    """Initializes agents, registers tools, and starts the conversation."""
    logger.info("--- Starting Multi-Agent Chat ---")

    # --- Instantiate Agents ---
    try:
        memory_agent = PineconeMemoryAgent(
            name="memory_manager",
            pinecone_index_instance=pinecone_index, 
            embedding_client_instance=openai_client 
        )
        executor_agent = create_executor_agent()
        code_gen_agent = create_code_generator_agent()
        planner_agent = create_planner_agent(
            memory_agent_name=memory_agent.name, 
            code_gen_name=code_gen_agent.name,
            exec_name=executor_agent.name
        )
        user_proxy = create_user_proxy_agent()

    except Exception as e:
        logger.critical(f"Failed to instantiate agents: {e}", exc_info=True)
        print(f"\n🚨 Error: Could not create necessary agents. Check logs. Error: {e}")
        sys.exit(1)


    # --- Register Planner Tools ---
    try:
        function_map = {
            "store_texts_in_memory": functools.partial(
                tools.store_texts_in_memory,
                planner_agent=planner_agent,
                memory_agent=memory_agent
            ),
            "retrieve_texts_from_memory": functools.partial(
                tools.retrieve_texts_from_memory,
                planner_agent=planner_agent,
                memory_agent=memory_agent
            ),
            "generate_python_code": functools.partial(
                tools.generate_python_code,
                planner_agent=planner_agent,
                code_generator_agent=code_gen_agent
            ),
            "execute_python_code": functools.partial(
                tools.execute_python_code,
                planner_agent=planner_agent,
                executor_agent=executor_agent
            ),
        }
        planner_agent.register_function(function_map=function_map)
        logger.info("Successfully registered tool functions with the Planner agent.")
    except Exception as e:
        logger.critical(f"Failed to register tool functions: {e}", exc_info=True)
        print(f"\n🚨 Error: Could not register tools for the Planner agent. Check logs. Error: {e}")
        sys.exit(1)

    # --- Display System Info ---
    print(f"\n🤖 AutoGen Multi-Agent System Initialized 🤖")
    print(f"------------------------------------------")
    print(f"Planner LLM:        {config.PLANNER_LLM_NAME}")
    print(f"Code Generator LLM: {config.CODE_GEN_LLM_NAME}")
    print(f"Memory Agent:       {memory_agent.name} (Pinecone Index: {config.ENV_VARS['PINECONE_INDEX_NAME']})")
    print(f"Executor Agent:     {executor_agent.name} (./{config.CODE_EXECUTION_DIR})")
    print(f"------------------------------------------")
    print(f"Date/Time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Keys loaded via .env file.")
    print("------------------------------------------")
    print(f"\nEnter your request for the planner. Type 'exit', 'quit', 'stop', or 'goodbye' to end the session.\n")


    # --- Initiate Chat ---
    try:
        user_proxy.initiate_chat(
            recipient=planner_agent,
            message=None, # Let user provide the first message via input prompt
            clear_history=True,
        )
    except Exception as e:
        logger.critical(f"An error occurred during the chat execution: {e}", exc_info=True)
        print(f"\n🚨 An error occurred during the chat. Check logs. Error: {e}")


if __name__ == "__main__":
    try:
        run_chat()
    except ValueError as ve:
        logger.critical(f"Configuration error: {ve}", exc_info=True)
        print(f"\n🚨 Configuration Error: {ve}. Please check your .env file and setup.")
    except RuntimeError as rte:
        logger.critical(f"Runtime error: {rte}", exc_info=True)
        print(f"\n🚨 Runtime Error: {rte}. Please check external service connections and logs.")
    except ImportError as ie:
         logger.critical(f"Import error: {ie}. Make sure all dependencies are installed (`pip install -r requirements.txt`).", exc_info=True)
         print(f"\n🚨 Import Error: {ie}. Please install required libraries using `pip install -r requirements.txt`.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred in main: {e}", exc_info=True)
        print(f"\n🚨 An unexpected error occurred. Check logs. Error: {e}")
    finally:
        print("\n👋 Session terminated.")
        logging.shutdown() 