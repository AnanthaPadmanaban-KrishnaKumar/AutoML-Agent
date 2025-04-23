# services.py
import logging
import pinecone
from openai import OpenAI, APIError
from typing import Optional

# Import specific config values needed
from config import ENV_VARS, logger

def initialize_pinecone() -> Optional[pinecone.Index]:
    """Initializes connection to Pinecone and returns the Index object."""
    api_key = ENV_VARS.get("PINECONE_API_KEY")
    environment = ENV_VARS.get("PINECONE_ENVIRONMENT")
    index_name = ENV_VARS.get("PINECONE_INDEX_NAME")

    if not all([api_key, environment, index_name]):
        logger.error("Pinecone configuration missing in environment variables.")
        return None

    try:
        pinecone.init(api_key=api_key, environment=environment)
        if index_name not in pinecone.list_indexes():
            logger.warning(f"Pinecone index '{index_name}' not found. Please create it.")
            # Depending on requirements, you might raise an error or return None
            # raise ValueError(f"Pinecone index '{index_name}' not found.")
            return None # Or handle index creation if desired/safe

        index = pinecone.Index(index_name)
        stats = index.describe_index_stats()
        logger.info(f"Successfully connected to Pinecone index '{index_name}'. Stats: {stats}")
        return index
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone or connect to index '{index_name}': {e}", exc_info=True)
        return None # Or raise the exception: raise

def initialize_openai() -> Optional[OpenAI]:
    """Initializes the OpenAI client."""
    api_key = ENV_VARS.get("OPENAI_API_KEY")
    if not api_key:
         logger.error("OpenAI API key not found in environment variables.")
         return None
    try:
        client = OpenAI(api_key=api_key)
        client.models.list() # Test connection
        logger.info("OpenAI client initialized successfully.")
        return client
    except APIError as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        return None # Or raise the exception: raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during OpenAI initialization: {e}", exc_info=True)
        return None # Or raise

# Initialize services on import
pinecone_index: Optional[pinecone.Index] = initialize_pinecone()
openai_client: Optional[OpenAI] = initialize_openai()

# Perform a hard check after initialization attempt
if pinecone_index is None or openai_client is None:
    raise RuntimeError("Failed to initialize essential services (Pinecone or OpenAI). Check logs and .env file.")