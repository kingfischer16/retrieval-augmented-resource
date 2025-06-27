"""
config.py
#########

Basic static configuration for the application, including model names and vector store settings.

"""


# Model names to use
GEMINI_EMBEDDING_MODEL_NAME = "gemini-embedding-001"
GEMINI_CHAT_MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"

# Model options
CHAT_MODEL_TEMPERATEURE = 0.2  # Temperature for the chat model, controlling randomness in responses
CHAT_MODEL_THINKING_BUDGET = 0  # Thinking budget for the chat model, controlling the amount of computation used for generating responses

# Vector store settings
CHUNK_SIZE = 1000  # Default chunk size for text splitting
CHUNK_OVERLAP = 200  # Default chunk overlap for text splitting
