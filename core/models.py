"""
models.py
#########

Contains the LLM and embedding model names to be used in the application, as well as 
instantiates the models so that they can be imported and used in other parts of the application.

Also contains the model options.
"""

# Imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from core.config import (
    GEMINI_EMBEDDING_MODEL_NAME, 
    GEMINI_CHAT_MODEL_NAME,
    CHAT_MODEL_TEMPERATURE,
    CHAT_MODEL_THINKING_BUDGET
)

# Instantiate the chat model
# This model is used for generating chat responses based on user input
chat_model = ChatGoogleGenerativeAI(
    model=GEMINI_CHAT_MODEL_NAME, 
    temperature=CHAT_MODEL_TEMPERATURE, 
    thinking_budget=CHAT_MODEL_THINKING_BUDGET, 
    verbose=False
)

# Instanitate the embedding model
# This model is used to convert text into embeddings for vector storage and similarity search
embedding_model = GoogleGenerativeAIEmbeddings(
    model=GEMINI_EMBEDDING_MODEL_NAME, 
    verbose=False
)
