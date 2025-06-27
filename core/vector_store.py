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
from langchain.tools.retriever import create_retriever_tool

# Custom imports
from core.utils import get_or_create_app_dir, add_vector_store_to_registry
from core.config import CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_RETRIEVAL_K
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

def build_document_list_from_folder(folder_path: str, max_files: int = None) -> tuple[bool, list]:
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
                print(f"Searching for files matching pattern: {pattern}")
                
                # First, get the list of files matching the pattern
                matching_files = list(folder.rglob(pattern.replace("**/", "")))
                
                if not matching_files:
                    continue
                
                # Apply file limit if specified
                if max_files and len(all_documents) >= max_files:
                    print(f"Reached maximum file limit ({max_files}), stopping file processing.")
                    break
                
                # Limit the files for this pattern if needed
                remaining_slots = max_files - len(all_documents) if max_files else len(matching_files)
                if max_files:
                    matching_files = matching_files[:remaining_slots]
                
                print(f"Found {len(matching_files)} files matching {pattern} (processing {len(matching_files)})")
                
                # Process files in smaller batches to avoid memory issues
                batch_size = 50  # Process 50 files at a time
                total_batches = (len(matching_files) + batch_size - 1) // batch_size
                
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(matching_files))
                    batch_files = matching_files[start_idx:end_idx]
                    
                    print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_files)} files)...")
                    
                    # Process each file individually to avoid memory issues
                    batch_documents = []
                    for file_path in batch_files:
                        try:
                            # Use the individual file loader for better control
                            success, file_docs = build_document_list_from_file(str(file_path))
                            if success and file_docs:
                                batch_documents.extend(file_docs)
                            
                            # Check if we've reached the file limit
                            if max_files and len(all_documents) + len(batch_documents) >= max_files:
                                print(f"Reached maximum file limit ({max_files}), stopping.")
                                break
                                
                        except Exception as e:
                            print(f"Warning: Failed to load {file_path}: {str(e)}")
                            continue
                    
                    if batch_documents:
                        all_documents.extend(batch_documents)
                        print(f"Loaded {len(batch_documents)} documents from batch {batch_idx + 1} (total: {len(all_documents)})")
                    
                    # Check if we've reached the limit
                    if max_files and len(all_documents) >= max_files:
                        break
                    
                    # Optional: Add a small delay to prevent overwhelming the system
                    import time
                    time.sleep(0.1)
                
                # Break out of pattern loop if we've reached the limit
                if max_files and len(all_documents) >= max_files:
                    break
                        
            except Exception as e:
                print(f"Warning: Failed to process files matching pattern '{pattern}': {str(e)}")
                continue
        
        if not all_documents:
            print(f"No supported documents found in '{folder_path}' or its subdirectories.")
            return False, []
        
        print(f"Total documents loaded: {len(all_documents)}")
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

def build_document_list(file_or_folder_path: str, max_files: int = None) -> tuple[bool, list]:
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
        return build_document_list_from_folder(file_or_folder_path, max_files)
    else:
        print(f"Error: The path '{file_or_folder_path}' is neither a file nor a directory.")
        return False, []

def create_vector_store(
    file_or_folder_path: str, 
    store_name: str,
    description: str,
    collection_name: str,
    text_splitter: Literal["recursive", "token"] = "recursive",
    debug: bool = False,
    max_files: int = None
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
    print("Step 1: Loading documents...")
    try:
        success, documents = build_document_list(file_or_folder_path, max_files)
    except Exception as e:
        print(f"Error during document loading: {str(e)}")
        return
    
    if not success or not documents:
        print(f"Error: Failed to load documents from '{file_or_folder_path}'. Vector store creation aborted.")
        return
    
    print(f"Loaded {len(documents)} documents successfully.")
    
    # Check if we have too many documents and warn the user
    if len(documents) > 1000:
        print(f"Warning: Processing {len(documents)} documents. This may take a while and use significant API quota.")
        print("Consider processing smaller batches or using a subset of files.")
    
    # Step 2: Set up text splitter based on user choice
    print("Step 2: Setting up text splitter...")
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
    print(f"Step 3: Splitting documents using '{text_splitter}' splitter...")
    try:
        texts = splitter.split_documents(documents)
        print(f"Created {len(texts)} text chunks from {len(documents)} documents.")
        
        # Warn about large chunk counts
        if len(texts) > 10000:
            print(f"Warning: {len(texts)} chunks will be processed. This may take a very long time.")
            print("Consider using larger chunk sizes or processing fewer documents.")
            
    except Exception as e:
        print(f"Error: Failed to split documents: {e}")
        return
    
    # Step 4: Set up embeddings using the custom embedding model
    print("Step 4: Setting up embeddings...")
    try:
        embeddings = embedding_model
    except Exception as e:
        print(f"Error: Failed to initialize embeddings: {e}")
        return
    
    # Step 5: Create persist directory
    print("Step 5: Creating persist directory...")
    try:
        cache_dir = get_or_create_cache_dir()
        persist_directory = str(Path(cache_dir) / "chroma_db" / collection_name)
        
        # Create the directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        print(f"Vector store will be persisted to: {persist_directory}")
    except Exception as e:
        print(f"Error: Failed to create persist directory: {e}")
        return
    
    # Step 6: Create vector store
    print("Step 6: Creating vector store with embeddings...")
    print("This may take a while depending on the number of chunks and API rate limits...")
    try:
        # Process in smaller batches to avoid overwhelming the API
        batch_size = 100  # Process 100 chunks at a time
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        if len(texts) > batch_size:
            print(f"Processing {len(texts)} chunks in {total_batches} batches of {batch_size}...")
            
            # Create vector store with first batch
            first_batch = texts[:batch_size]
            vector_store = Chroma.from_documents(
                documents=first_batch,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            print(f"Processed batch 1/{total_batches} ({len(first_batch)} chunks)")
            
            # Add remaining batches
            for i in range(1, total_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(texts))
                batch = texts[start_idx:end_idx]
                
                print(f"Processing batch {i+1}/{total_batches} ({len(batch)} chunks)...")
                vector_store.add_documents(batch)
                print(f"Completed batch {i+1}/{total_batches}")
        else:
            # Small number of documents, process all at once
            vector_store = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        
        print(f"Vector store created successfully!")
        
    except Exception as e:
        print(f"Error: Failed to create vector store: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return
    
    # Step 7: Register the vector store in the registry
    print("Registering vector store in registry...")
    success = add_vector_store_to_registry(
        name=store_name,
        collection_name=collection_name,
        persist_directory=persist_directory,
        description=description,
        debug=debug
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
        search_kwargs = {"k": DEFAULT_RETRIEVAL_K}  # Default to returning top documents from config
    
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

def create_retriever_tool_from_vector_store(
    collection_name: str,
    persist_directory: str,
    store_name: str,
    search_type: str = "similarity",
    search_kwargs: dict = None
) -> object:
    """
    Create a retriever tool from a vector store that an agent can use.
    
    This function gets a retriever from the vector store and wraps it as a tool
    that can be used by LangChain agents.
    
    Args:
        collection_name: The name of the collection to retrieve from
        persist_directory: The directory where the vector store is persisted
        store_name: The name of the vector store for tool description
        search_type: The type of search to perform
        search_kwargs: Additional search parameters
        
    Returns:
        A LangChain tool that the agent can use to search the vector store
    """
    from core.prompts import RETRIEVER_TOOL_DESCRIPTION
    
    # Get the retriever
    retriever = get_retriever_from_vector_store(
        collection_name=collection_name,
        persist_directory=persist_directory,
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    # Create and return the tool
    return create_retriever_tool(
        retriever=retriever,
        name="search_knowledge_base",
        description=f"Search the {store_name} knowledge base. {RETRIEVER_TOOL_DESCRIPTION}"
    )
