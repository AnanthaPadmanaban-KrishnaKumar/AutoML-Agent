import logging
import time 
import pinecone
from openai import OpenAI, APIError
from typing import Optional
from config import ENV_VARS, logger, DEFAULT_PINECONE_METRIC

# --- Pinecone Initialization ---
def initialize_pinecone() -> Optional[pinecone.Index]:
    """
    Initializes connection to Pinecone.
    Checks if the index exists, and if not, attempts to create it.
    Returns the Index object or None on failure.
    """
    # --- Get Pinecone Configuration ---
    api_key = ENV_VARS.get("PINECONE_API_KEY")
    environment = ENV_VARS.get("PINECONE_ENVIRONMENT")
    index_name = ENV_VARS.get("PINECONE_INDEX_NAME")
    # <<< START GETTING NEW CONFIG FOR CREATION >>>
    dimension_str = ENV_VARS.get("PINECONE_VECTOR_DIMENSION") 
    metric = ENV_VARS.get("PINECONE_METRIC", DEFAULT_PINECONE_METRIC) 
    # <<< END GETTING NEW CONFIG FOR CREATION >>>

    # --- Basic Configuration Check ---
    # Dimension string must now also be present 
    if not all([api_key, environment, index_name, dimension_str]):
        missing = [k for k,v in {
            "Pinecone API Key": api_key,
            "Pinecone Environment": environment,
            "Pinecone Index Name": index_name,
            "Pinecone Vector Dimension": dimension_str
        }.items() if not v]
        logger.error(f"Pinecone configuration incomplete. Cannot proceed. Missing: {', '.join(missing)}.")
        return None

    # --- Convert dimension to integer ---
    try:
        dimension = int(dimension_str)
    except (ValueError, TypeError): 
         logger.error(f"Internal Error: Invalid PINECONE_VECTOR_DIMENSION encountered: '{dimension_str}'. Must be an integer.")
         return None

    # --- Initialize Pinecone connection ---
    try:
        logger.info(f"Initializing Pinecone connection for environment '{environment}'...")
        pinecone.init(api_key=api_key, environment=environment)
        logger.info("Pinecone connection initialized.")

        # --- Check if index exists ---
        if index_name not in pinecone.list_indexes():
            logger.warning(f"Pinecone index '{index_name}' not found. Attempting to create...")

            # <<< START INDEX CREATION BLOCK >>>
            try:
                pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric
                )
                wait_time = 15
                logger.info(f"Index '{index_name}' created successfully with dimension {dimension} and metric '{metric}'. Waiting {wait_time} seconds for it to initialize...")
                time.sleep(wait_time)
            except Exception as create_e:
                logger.error(f"Failed to create Pinecone index '{index_name}': {create_e}", exc_info=True)
                return None # Return None if creation fails
            # <<< END INDEX CREATION BLOCK >>>
        else:
            logger.info(f"Pinecone index '{index_name}' already exists.")

        # --- Index exists or was just created, get the Index object ---
        logger.info(f"Connecting to Pinecone index '{index_name}'...")
        index = pinecone.Index(index_name)
        try:
             stats = index.describe_index_stats()
             logger.info(f"Successfully connected to Pinecone index '{index_name}'. Stats: {stats}")
             return index
        except Exception as conn_e:
             logger.error(f"Connected to Pinecone, but failed to get stats for index '{index_name}': {conn_e}", exc_info=True)
             return None 

    except Exception as e:
        logger.error(f"Failed during Pinecone initialization or connection attempt for index '{index_name}': {e}", exc_info=True)
        return None

# --- OpenAI Initialization ---
def initialize_openai() -> Optional[OpenAI]:
    """Initializes the OpenAI client (needed for Embeddings at minimum)."""
    api_key = ENV_VARS.get("OPENAI_API_KEY")
    if not api_key:
         logger.error("OpenAI API key not found in environment variables.")
         return None
    try:
        logger.info("Initializing OpenAI client...")
        client = OpenAI(api_key=api_key)
        client.models.list()
        logger.info("OpenAI client initialized successfully.")
        return client
    except APIError as e:
        logger.error(f"OpenAI API Error during client initialization: {e}", exc_info=False)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during OpenAI initialization: {e}", exc_info=True)
        return None

# --- Initialize services on import ---
# Attempt to initialize both services
pinecone_index: Optional[pinecone.Index] = initialize_pinecone()
openai_client: Optional[OpenAI] = initialize_openai()

# --- Perform a hard check after initialization attempts ---
# If either essential service failed, stop execution.
if pinecone_index is None or openai_client is None:
    failed_services = []
    if pinecone_index is None: failed_services.append("Pinecone")
    if openai_client is None: failed_services.append("OpenAI (for embeddings)")
    error_message = f"Failed to initialize essential services ({', '.join(failed_services)}). Check logs and .env file configuration."
    logger.critical(error_message)
    raise RuntimeError(error_message)
else:
    logger.info("Essential services (Pinecone Index, OpenAI Client) initialized successfully.")