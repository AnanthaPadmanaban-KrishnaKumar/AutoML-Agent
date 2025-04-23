import os
import logging
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI, APIError

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
EMBEDDING_MODEL = "text-embedding-ada-002"


PINECONE_TOP_K = 5
CODE_EXECUTION_DIR = "coding"
CODE_EXEC_TIMEOUT = 120

# LLM Model Identifiers (Use the official identifiers)
# Planner options
GPT4O_MODEL = "gpt-4o"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
CLAUDE_SONNET_MODEL = "claude-3-5-sonnet-20240620"


# --- Logging Setup ---
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a root logger."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S' 
    )

    logger = logging.getLogger("ConfigSetup") 
    logger.setLevel(level) 
    return logger

# Initialize logger for setup messages
logger = setup_logging()

# --- Environment Variable Loading & Validation ---
def load_and_validate_envs() -> Dict[str, Optional[str]]:
    """
    Loads environment variables from .env and validates them based on
    strict requirements for Planner, Code Generator, Embeddings, and Memory.
    """
    logger.info("Loading environment variables from .env file...")
    envs = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT"),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
    }
    logger.info("Environment variables loaded. Starting validation...")

    # --- Validation based on agent requirements ---
    errors = []

    # 1. Pinecone (Memory)
    pinecone_keys = ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "PINECONE_INDEX_NAME"]
    missing_pinecone = [k for k in pinecone_keys if not envs[k]]
    if missing_pinecone:
        errors.append(f"Missing Pinecone configuration: {', '.join(missing_pinecone)}")

    # 2. OpenAI (Embeddings are always needed)
    if not envs["OPENAI_API_KEY"]:
        errors.append("Missing OPENAI_API_KEY (required for text embeddings)")

    # 3. Anthropic (Code Generator - strictly required)
    if not envs["ANTHROPIC_API_KEY"]:
        errors.append(f"Missing ANTHROPIC_API_KEY (required for Code Generator using {CLAUDE_SONNET_MODEL})")

    # 4. Planner (Need at least one: OpenAI or DeepSeek)
    planner_keys_present = any([envs["OPENAI_API_KEY"], envs["DEEPSEEK_API_KEY"]])
    if not planner_keys_present:
        errors.append("Planner requires at least OPENAI_API_KEY or DEEPSEEK_API_KEY to be set")

    # --- Report errors or confirm success ---
    if errors:
        error_msg = ("Configuration validation failed. Ensure required variables "
                     "are set correctly in your .env file:\n- " +
                     "\n- ".join(errors))
        logger.error(error_msg)
        raise ValueError(error_msg) # Stop execution if validation fails
    else:
        logger.info("Environment variables successfully validated.")
        if envs["DEEPSEEK_API_KEY"]:
            logger.info("DEEPSEEK_API_KEY found (available as Planner fallback).")
        else:
            logger.info("DEEPSEEK_API_KEY not found (Planner will rely solely on OpenAI).")

    return envs

# Load and validate environment variables on import
ENV_VARS = load_and_validate_envs()


# --- LLM Configurations ---
def get_llm_configs(env_vars: Dict[str, Optional[str]]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Builds LLM configuration dictionaries based *only* on validated, available API keys."""
    logger.info("Preparing LLM configurations...")
    configs = {
        "openai_gpt4o": None,
        "claude_sonnet": None,
        "deepseek": None
    }

    # OpenAI Config
    if env_vars["OPENAI_API_KEY"]:
        configs["openai_gpt4o"] = {
            "model": GPT4O_MODEL,
            "api_key": env_vars["OPENAI_API_KEY"],
            "temperature": 0.2, 
        }
        logger.info(f"OpenAI ({GPT4O_MODEL}) config prepared.")

    # Anthropic Config 
    if env_vars["ANTHROPIC_API_KEY"]:
        configs["claude_sonnet"] = {
            "model": CLAUDE_SONNET_MODEL,
            "api_key": env_vars["ANTHROPIC_API_KEY"],
            "temperature": 0.1, 
        }
        logger.info(f"Anthropic ({CLAUDE_SONNET_MODEL}) config prepared.")

    # DeepSeek Config 
    if env_vars["DEEPSEEK_API_KEY"]:
        configs["deepseek"] = {
            "model": DEEPSEEK_CHAT_MODEL,
            "api_key": env_vars["DEEPSEEK_API_KEY"],
            "base_url": DEEPSEEK_BASE_URL,
            "temperature": 0.2,
            "api_type": "openai", 
        }
        logger.info(f"DeepSeek ({DEEPSEEK_CHAT_MODEL}) config prepared.")

    logger.info("LLM configurations preparation complete.")
    return configs

# Prepare configurations based on validated environment variables
LLM_CONFIGS = get_llm_configs(ENV_VARS)


# --- Select Specific LLM Configurations for Agent Roles ---
def select_agent_llms(
    available_configs: Dict[str, Optional[Dict[str, Any]]]
) -> Tuple[Dict[str, Any], str, Dict[str, Any], str]:
    """
    Selects the final LLM configurations for the Planner and Code Generator agents
    based on the enforced requirements.
    """
    logger.info("Selecting LLM configurations for specific agent roles...")

    # --- Planner Agent Selection: OpenAI GPT-o3 OR DeepSeek R1 ---
    if available_configs["openai_gpt4o"]:
        planner_config = available_configs["openai_gpt4o"]
        planner_name = f"OpenAI {GPT4O_MODEL}"
        logger.info(f"Planner Agent will use primary choice: {planner_name}")
    elif available_configs["deepseek"]:
        planner_config = available_configs["deepseek"]
        planner_name = f"DeepSeek {DEEPSEEK_CHAT_MODEL}"
        logger.info(f"Planner Agent will use fallback: {planner_name}")
    else:
        logger.critical("PANIC: No valid configuration found for Planner Agent despite passing validation.")
        raise RuntimeError("Configuration error: No LLM available for Planner.")

    # --- Code Generator Agent Selection: Claude Sonnet ---
    if available_configs["claude_sonnet"]:
        code_gen_config = available_configs["claude_sonnet"]
        code_gen_name = f"Claude {CLAUDE_SONNET_MODEL}"
        logger.info(f"Code Generator Agent will use required choice: {code_gen_name}")
    else:
        logger.critical(f"PANIC: Required Claude Sonnet ({CLAUDE_SONNET_MODEL}) config not found despite passing validation.")
        raise RuntimeError(f"Configuration error: Required LLM ({CLAUDE_SONNET_MODEL}) for Code Generator not available.")

    logger.info("Agent LLM selection complete.")
    return planner_config, planner_name, code_gen_config, code_gen_name

# Assign the selected configurations and names to be exported
PLANNER_LLM_CONFIG, PLANNER_LLM_NAME, CODE_GEN_LLM_CONFIG, CODE_GEN_LLM_NAME = select_agent_llms(LLM_CONFIGS)


# --- Initialize and Export OpenAI Client (for Embeddings) ---
try:
    logger.info(f"Initializing OpenAI client for embeddings (using model: {EMBEDDING_MODEL})...")
    OPENAI_CLIENT = OpenAI(api_key=ENV_VARS["OPENAI_API_KEY"])
    OPENAI_CLIENT.models.list(limit=1)
    logger.info("OpenAI client initialized successfully for embeddings.")
except APIError as e:
    logger.error(f"OpenAI API Error during client initialization (check API key and network): {e}", exc_info=False)
    raise RuntimeError(f"Failed to initialize OpenAI client needed for embeddings: {e}") from e
except Exception as e:
    logger.error(f"An unexpected error occurred during OpenAI client initialization: {e}", exc_info=True) 
    raise RuntimeError(f"Unexpected error initializing OpenAI client: {e}") from e


# --- Final Check and Export Summary ---
logger.info("Configuration loading and setup complete.")