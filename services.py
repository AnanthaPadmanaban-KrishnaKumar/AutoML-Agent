# services.py

import logging
import time
from pinecone import Pinecone, ServerlessSpec, Index # Updated Pinecone imports
from openai import OpenAI, APIError
from typing import Optional
from config import ENV_VARS, logger, DEFAULT_PINECONE_METRIC, DEFAULT_PINECONE_CLOUD # Import defaults

# --- Pinecone Initialization ---
def initialize_pinecone() -> Optional[Index]: # Return type uses imported Index
    """
    Initializes connection to Pinecone using the new client pattern.
    Checks if the index exists, and if not, attempts to create it using ServerlessSpec.
    Returns the Index object or None on failure.
    """
    # --- Get Pinecone Configuration ---
    api_key = ENV_VARS.get("PINECONE_API_KEY")
    environment = ENV_VARS.get("PINECONE_ENVIRONMENT") # Used as region for ServerlessSpec
    index_name = ENV_VARS.get("PINECONE_INDEX_NAME")
    dimension_str = ENV_VARS.get("PINECONE_VECTOR_DIMENSION") # Already validated/defaulted in config.py
    metric = ENV_VARS.get("PINECONE_METRIC", DEFAULT_PINECONE_METRIC) # Use default if needed
    cloud = ENV_VARS.get("PINECONE_CLOUD", DEFAULT_PINECONE_CLOUD) # Get cloud provider, use default

    # --- Basic Configuration Check ---
    if not all([api_key, environment, index_name, dimension_str, cloud]):
        missing = [k for k,v in {
            "Pinecone API Key": api_key,
            "Pinecone Environment/Region": environment,
            "Pinecone Index Name": index_name,
            "Pinecone Vector Dimension": dimension_str,
            "Pinecone Cloud": cloud
        }.items() if not v]
        logger.error(f"Pinecone configuration incomplete. Cannot proceed. Missing: {', '.join(missing)}.")
        return None

    # --- Convert dimension to integer ---
    try:
        dimension = int(dimension_str)
    except (ValueError, TypeError): # Should not happen if config validation passed, but good practice
         logger.error(f"Internal Error: Invalid PINECONE_VECTOR_DIMENSION encountered: '{dimension_str}'. Must be an integer.")
         return None

    # --- Initialize Pinecone Client ---
    try:
        logger.info(f"Initializing Pinecone client...")
        pc = Pinecone(api_key=api_key) # Instantiate the client
        logger.info(f"Pinecone client initialized. Checking index '{index_name}'...")

        # --- Check if index exists using the new client ---
        if index_name not in pc.list_indexes().names(): # Use client method and access .names attribute
            logger.warning(f"Pinecone index '{index_name}' not found. Attempting to create with ServerlessSpec...")

            # --- Define Serverless Spec ---
            try:
                spec = ServerlessSpec(cloud=cloud, region=environment) # Use environment as region
            except Exception as spec_e: # Catch potential errors in spec definition (e.g., invalid region/cloud)
                 logger.error(f"Failed to create Pinecone ServerlessSpec(cloud='{cloud}', region='{environment}'): {spec_e}", exc_info=True)
                 return None

            # --- Attempt Index Creation ---
            try:
                pc.create_index( # Use client method
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=spec
                )
                wait_time = 30 # Serverless might take a bit longer initially
                logger.info(f"Index '{index_name}' creation initiated successfully with spec {spec}. Waiting {wait_time} seconds for provisioning...")
                time.sleep(wait_time)
                # Optional: Add a loop here to check pc.describe_index(index_name).status == 'Ready' for robustness
            except Exception as create_e:
                logger.error(f"Failed to create Pinecone index '{index_name}': {create_e}", exc_info=True)
                return None # Return None if creation fails

        else:
            logger.info(f"Pinecone index '{index_name}' already exists.")

        # --- Index exists or was just created, get the Index object via client ---
        logger.info(f"Connecting to Pinecone index '{index_name}'...")
        index: Index = pc.Index(index_name) # Get the Index object using the client

        # --- Verify connection by getting stats ---
        try:
             # Give it a few seconds to ensure readiness after creation or connection
             time.sleep(5)
             stats = index.describe_index_stats() # Use the Index object method
             logger.info(f"Successfully connected to Pinecone index '{index_name}'. Stats: {stats}")
             return index # Return the Index object
        except Exception as conn_e:
             logger.error(f"Connected to Pinecone, but failed to get stats for index '{index_name}'. It might still be initializing: {conn_e}", exc_info=True)
             return None # Fail if we can't interact with the index

    except Exception as e:
        # Catch errors during client init or other operations
        logger.error(f"Failed during Pinecone client initialization or index handling for '{index_name}': {e}", exc_info=True)
        return None

# --- OpenAI Initialization ---
def initialize_openai() -> Optional[OpenAI]:
    """Initializes the OpenAI client (needed for Embeddings at minimum)."""
    api_key = ENV_VARS.get("OPENAI_API_KEY")
    if not api_key:
         # This case is already checked in config validation, but double-check here.
         logger.error("OpenAI API key not found in environment variables.")
         return None
    try:
        logger.info("Initializing OpenAI client...")
        client = OpenAI(api_key=api_key)
        # Perform a simple API call to verify the key and connection
        client.models.list()
        logger.info("OpenAI client initialized successfully.")
        return client
    except APIError as e:
        # Specific handling for API errors (e.g., invalid key, rate limits)
        logger.error(f"OpenAI API Error during client initialization: {e}", exc_info=False) # Less verbose log for API keys
        return None
    except Exception as e:
        # Catch other potential issues (network errors, etc.)
        logger.error(f"An unexpected error occurred during OpenAI initialization: {e}", exc_info=True)
        return None

# --- Initialize services on import ---
# Attempt to initialize both services
# Note: The type hint for pinecone_index remains Optional[Index] from pinecone library
pinecone_index: Optional[Index] = initialize_pinecone()
openai_client: Optional[OpenAI] = initialize_openai()

# --- Perform a hard check after initialization attempts ---
# If either essential service failed, stop execution.
if pinecone_index is None or openai_client is None:
    failed_services = []
    if pinecone_index is None: failed_services.append("Pinecone")
    if openai_client is None: failed_services.append("OpenAI (for embeddings)")
    error_message = f"Failed to initialize essential services ({', '.join(failed_services)}). Check logs and .env file configuration."
    logger.critical(error_message)
    # Raising RuntimeError will be caught by the main script's exception handler
    raise RuntimeError(error_message)
else:
    logger.info("Essential services (Pinecone Index, OpenAI Client) initialized successfully.")