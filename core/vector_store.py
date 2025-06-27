"""
create_vector_store.py
######################

This module contains the functionality to create a new vector store from a given file or folder path.

It is expected that the user will provide either a valid file path or a valid folder path. Invalid files or 
folders will result in an error message and the user will be prompted to try again.

It is expected that the user will provide a file in one of the following formats: PDF, TXT, MD, CSV, or HTML.

If the user provides a folder path, the module will assume that the user is only interested in the 
following file formats: PDF, TXT, MD, CSV, HTML.

If the provided folder path contains more than one of the acceptable file formats, the module will recognize this and 
read all relevant files in the folder using the appropriate document loaders, and create a single collection
of documents that will be used to create the vector store.


"""
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Literal

# Langchain imports
from langchain.document_loaders import (
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    PyPDFLoader,
    DirectoryLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, 
    TokenTextSplitter
)
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.vectorstores import Chroma

# Custom imports
from core.utils import get_or_create_app_dir, add_vector_store_to_registry
from core.config import CHUNK_SIZE, CHUNK_OVERLAP
from core.models import embedding_model

# Constants
APP_DATA_FOLDER = "cache"

# Helper functions
def get_or_create_cache_dir() -> str:
    """
    Gets the data cache directory for the application, creating it if it does not exist.

    Returns:
        str: The path to data cache directory.
    """
    data_dir = Path(get_or_create_app_dir()) / APP_DATA_FOLDER
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created data cache directory at: {data_dir}")
    else:
        print(f"Data cache directory already exists at: {data_dir}")
    return str(data_dir)

def build_document_list_from_folder(folder_path: str) -> tuple[bool, list]:
    """
    Build a list of documents from a folder path using DirectoryLoader. 

    This function will accept the folder path as provided by the user. First, it will check if the path is a valid directory.
    If it is not, it will return False and an empty list.

    If the path is valid, it will use DirectoryLoader to recursively load all supported files in the directory 
    and its subdirectories. The DirectoryLoader automatically handles multiple file types including: 
    PDF, TXT, MD, CSV, HTML, and other formats supported by LangChain.

    The function will load all documents from the directory tree and return them as a single collection
    ready for chunking and vectorization.

    Args:
        folder_path (str): The path to the folder containing one or more files.

    Returns:
        tuple[bool, list]: A tuple containing:
            - bool: True if the folder path is valid and documents were processed, False otherwise
            - list: A list of loaded Document objects (empty if bool is False)
    """
    # Check if the provided path is a valid directory
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: The path '{folder_path}' does not exist.")
        return False, []
    
    if not folder.is_dir():
        print(f"Error: The path '{folder_path}' is not a directory.")
        return False, []
    
    try:
        # Use DirectoryLoader to recursively load all supported documents
        # Define glob patterns for supported file types
        glob_patterns = [
            "**/*.pdf", "**/*.txt", "**/*.md", "**/*.csv", 
            "**/*.html", "**/*.htm", "**/*.json", "**/*.docx"
        ]
        
        all_documents = []
        
        # Load documents for each file pattern
        for pattern in glob_patterns:
            try:
                loader = DirectoryLoader(
                    path=str(folder),
                    glob=pattern,
                    recursive=True,
                    show_progress=True,
                    use_multithreading=True
                )
                documents = loader.load()
                if documents:
                    all_documents.extend(documents)
                    print(f"Loaded {len(documents)} document(s) matching pattern '{pattern}'")
            except Exception as e:
                print(f"Warning: Failed to load files matching pattern '{pattern}': {str(e)}")
                continue
        
        if not all_documents:
            print(f"No supported documents found in '{folder_path}' or its subdirectories.")
            return False, []
        
        print(f"Total documents loaded recursively: {len(all_documents)}")
        return True, all_documents
        
    except Exception as e:
        print(f"Error: Failed to load documents from '{folder_path}': {str(e)}")
        return False, []

def build_document_list_from_file(file_path: str) -> tuple[bool, list]:
    """
    Build a list of documents from a file path using appropriate document loaders.

    This function will accept the file path as provided by the user. First, it will check if the path is a valid file.
    If it is not, it will return False and an empty list.

    If the path is valid, it will use the appropriate document loader based on the file extension to load the document.
    Supports PDF, TXT, MD, CSV, HTML, JSON, and DOCX files.

    Args:
        file_path (str): The path to the file to be loaded.

    Returns:
        tuple[bool, list]: A tuple containing:
            - bool: True if the file path is valid and documents were processed, False otherwise
            - list: A list of loaded Document objects (empty if bool is False)
    """
    # Check if the provided path is a valid file
    file = Path(file_path)
    if not file.exists():
        print(f"Error: The file '{file_path}' does not exist.")
        return False, []
    
    if not file.is_file():
        print(f"Error: The path '{file_path}' is not a file.")
        return False, []
    
    try:
        # Determine the file type and use the appropriate loader
        ext = file.suffix.lower()
        
        # Define supported loaders with their configurations
        loader_map = {
            ".pdf": lambda: PyPDFLoader(str(file)),
            ".txt": lambda: TextLoader(str(file), encoding="utf-8", autodetect_encoding=True),
            ".md": lambda: UnstructuredMarkdownLoader(str(file)),
            ".csv": lambda: CSVLoader(str(file), encoding="utf-8", autodetect_encoding=True),
            ".html": lambda: UnstructuredHTMLLoader(str(file)),
            ".htm": lambda: UnstructuredHTMLLoader(str(file)),
            ".json": lambda: TextLoader(str(file), encoding="utf-8", autodetect_encoding=True),
        }
        
        # Check if file type is supported
        if ext not in loader_map:
            supported_types = ", ".join(sorted(loader_map.keys()))
            print(f"Error: Unsupported file type '{ext}' for '{file_path}'. Supported types: {supported_types}")
            return False, []
        
        # Create and use the appropriate loader
        loader = loader_map[ext]()
        documents = loader.load()
        
        if not documents:
            print(f"Warning: No content could be extracted from '{file_path}'.")
            return False, []
        
        # Add source metadata if not already present
        for doc in documents:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = str(file)
            doc.metadata['file_type'] = ext
            doc.metadata['file_size'] = file.stat().st_size
        
        print(f"Successfully loaded {len(documents)} document(s) from '{file.name}'")
        return True, documents
        
    except Exception as e:
        print(f"Error: Failed to load document from '{file_path}': {str(e)}")
        return False, []

def build_document_list(file_or_folder_path: str) -> tuple[bool, list]:
    """
    Build a list of documents from a file or folder path as provided by the user.

    This function will determine if the provided path is a file or a folder.
    If it is a file, it will use the build_document_list_from_file function.
    If it is a folder, it will use the build_document_list_from_folder function.

    In either case, it will return a tuple containing a boolean indicating success or failure, as
    well as a list of Document objects that were loaded.

    Args:
        file_or_folder_path (str): The path to the file or folder to be loaded.

    Returns:
        tuple[bool, list]: A tuple containing:
            - bool: True if the path is valid and documents were processed, False otherwise
            - list: A list of loaded Document objects (empty if bool is False)
    """
    # Check if the provided path exists
    path = Path(file_or_folder_path)
    if not path.exists():
        print(f"Error: The path '{file_or_folder_path}' does not exist.")
        return False, []
    
    # Determine if the path is a file or folder and call appropriate function
    if path.is_file():
        print(f"Processing single file: {path.name}")
        return build_document_list_from_file(file_or_folder_path)
    elif path.is_dir():
        print(f"Processing folder: {path.name}")
        return build_document_list_from_folder(file_or_folder_path)
    else:
        print(f"Error: The path '{file_or_folder_path}' is neither a file nor a directory.")
        return False, []

def create_vector_store(
    file_or_folder_path: str, 
    store_name: str,
    description: str,
    collection_name: str,
    text_splitter: Literal["recursive", "token"] = "recursive"
    ) -> None:
    """
    Provided a file or folder path, this function will create a new vector store in the app data cache directory.

    This function gets the list of documents from the provided file or folder path using the build_document_list function.

    The collection name is will be created by the calling function, the store_name and description are provided by the user.
    the persist_directory will be create in this function based on the collection name.

    This function will overwrite an existing vector store if the same collection_name and persist_directory already exists.

    There is no return value for this function, but it will register the new vector store in the resigtry file. The calling
    function should then list all vector stores and the new store will be included in the list, ready to be chosen
    by the user for further operations.

    Args:
        file_or_folder_path (str): The path to the file or folder containing one or more files.
        store_name (str): The human-readable name of the vector store to be created.
        description (str): A description of the vector store.
        collection_name (str): The name of the collection to be created, used by Chroma.
        text_splitter (Literal["recursive", "token"]): The type of text splitter to use. Defaults to "recursive".
    """
    print(f"Creating vector store '{store_name}' from '{file_or_folder_path}'...")
    
    # Step 1: Load documents from the provided path
    success, documents = build_document_list(file_or_folder_path)
    
    if not success or not documents:
        print(f"Error: Failed to load documents from '{file_or_folder_path}'. Vector store creation aborted.")
        return
    
    print(f"Loaded {len(documents)} documents successfully.")
    
    # Step 2: Set up text splitter based on user choice
    if text_splitter == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    elif text_splitter == "token":
        splitter = TokenTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    else:
        print(f"Error: Unsupported text splitter '{text_splitter}'. Using 'recursive' as fallback.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
    
    # Step 3: Split documents into chunks
    print(f"Splitting documents using '{text_splitter}' splitter...")
    try:
        texts = splitter.split_documents(documents)
        print(f"Created {len(texts)} text chunks from {len(documents)} documents.")
    except Exception as e:
        print(f"Error: Failed to split documents: {e}")
        return
    
    # Step 4: Set up embeddings using the custom embedding model
    print("Setting up embeddings...")
    try:
        embeddings = embedding_model
    except Exception as e:
        print(f"Error: Failed to initialize embeddings: {e}")
        return
    
    # Step 5: Create persist directory
    cache_dir = get_or_create_cache_dir()
    persist_directory = str(Path(cache_dir) / "chroma_db" / collection_name)
    
    # Create the directory if it doesn't exist
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    print(f"Vector store will be persisted to: {persist_directory}")
    
    # Step 6: Create vector store
    print("Creating vector store with embeddings...")
    try:
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
        # Persist the vector store
        vector_store.persist()
        print(f"Vector store created and persisted successfully!")
        
    except Exception as e:
        print(f"Error: Failed to create vector store: {e}")
        return
    
    # Step 7: Register the vector store in the registry
    print("Registering vector store in registry...")
    success = add_vector_store_to_registry(
        name=store_name,
        collection_name=collection_name,
        persist_directory=persist_directory,
        description=description
    )
    
    if success:
        print(f"Vector store '{store_name}' successfully registered!")
        print(f"  - Name: {store_name}")
        print(f"  - Collection: {collection_name}")
        print(f"  - Documents: {len(documents)}")
        print(f"  - Text chunks: {len(texts)}")
        print(f"  - Splitter: {text_splitter}")
        print(f"  - Location: {persist_directory}")
        print("\nVector store is now available for selection in the main application.")
    else:
        print(f"Warning: Vector store created but failed to register in registry.")
        print("You may need to manually add it to the registry file.")

def get_retriever_from_vector_store(
    collection_name: str,
    persist_directory: str,
    search_type: str = "similarity",
    search_kwargs: dict = None
) -> VectorStoreRetriever:
    """
    Get a retriever from an existing vector store.

    This function loads an existing Chroma vector store from the specified persist directory
    and collection name, then returns a retriever that can be used in LangChain chains.

    Args:
        collection_name (str): The name of the collection to retrieve.
        persist_directory (str): The directory where the vector store is persisted.
        search_type (str): The type of search to perform. Options: "similarity", "mmr", "similarity_score_threshold".
        search_kwargs (dict): Additional search parameters. For similarity: {"k": int}, for mmr: {"k": int, "fetch_k": int, "lambda_mult": float}, for similarity_score_threshold: {"score_threshold": float, "k": int}.

    Returns:
        VectorStoreRetriever: The retriever for the specified vector store.

    Raises:
        FileNotFoundError: If the persist directory doesn't exist.
        ValueError: If the vector store cannot be loaded or is empty.
        Exception: For other errors during vector store loading.
    """
    # Validate persist directory exists
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        raise FileNotFoundError(f"Vector store directory does not exist: {persist_directory}")
    
    # Set default search kwargs if not provided
    if search_kwargs is None:
        search_kwargs = {"k": 4}  # Default to returning top 4 documents
    
    try:
        # Load the existing vector store with the same embedding model used during creation
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        
        # Verify the vector store has content
        try:
            # Quick test to see if the collection exists and has documents
            test_query = vector_store.similarity_search("test", k=1)
            if len(test_query) == 0:
                # Collection might be empty, but that's not necessarily an error
                print(f"Warning: Vector store '{collection_name}' appears to be empty.")
        except Exception:
            # If even a basic search fails, the vector store might be corrupted
            raise ValueError(f"Vector store '{collection_name}' appears to be corrupted or inaccessible.")
        
        # Create and return the retriever with specified search parameters
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        
        print(f"Successfully loaded retriever for vector store '{collection_name}'")
        return retriever
        
    except Exception as e:
        print(f"Error loading vector store '{collection_name}': {str(e)}")
        raise
