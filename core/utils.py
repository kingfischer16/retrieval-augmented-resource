"""
utils.py
########

Utility functions for the application, inlcuding setup and registry management for vector stores.
"""

# Imports
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from langchain_core.documents import Document

RAR_APP_NAME = "RetrievalAugmentedResource"

def get_app_dir() -> Path:
    """
    Get the cross-platform user-specific app directory.

    - Windows: C:\\Users\\<user>\\AppData\\Local\\<app_name>
    - macOS:   ~/Library/Application Support/<app_name>
    - Linux:   ~/.local/share/<app_name> or $XDG_DATA_HOME/<app_name>

    Returns:
        Path: The path to the app directory as a Path object.
    """
    # Check for Windows
    if sys.platform == "win32":
        # os.getenv('LOCALAPPDATA') is the correct variable for local app data.
        path = os.getenv('LOCALAPPDATA')
        if path is None:
            # Fallback for older systems or unusual configs, though LOCALAPPDATA is standard.
            path = os.path.expanduser("~")
        return Path(path) / RAR_APP_NAME
    
    # Check for macOS
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / RAR_APP_NAME
    
    # Check for Linux and other Unix-like systems
    else:
        # Follow the XDG Base Directory Specification
        xdg_data_home = os.getenv('XDG_DATA_HOME')
        if xdg_data_home:
            return Path(xdg_data_home) / RAR_APP_NAME
        else:
            # Default fallback if XDG_DATA_HOME is not set
            return Path.home() / ".local" / "share" / RAR_APP_NAME
        
def get_or_create_app_dir() -> str:
    """
    Gets the app directory for the application, creating it if it does not exist.

    Returns:
        str: The path to app directory.
    """
    app_dir = get_app_dir()
    if not app_dir.exists():
        app_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created app directory at: {app_dir}")
    else:
        print(f"App directory already exists at: {app_dir}")
    return str(app_dir)

# Vector Store Registry Functions
def get_vector_store_registry_path() -> Path:
    """
    Get the path to the vector store registry file, creating it if it doesn't exist.
    
    Returns:
        Path: The path to the registry JSON file.
    """
    registry_path = get_app_dir() / "vector_store_registry.json"
    
    # Create the registry file if it doesn't exist
    if not registry_path.exists():
        # Ensure the app directory exists first
        get_or_create_app_dir()
        
        # Create an empty registry file
        try:
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump({"stores": {}}, f, indent=2, ensure_ascii=False)
            print(f"Created vector store registry at: {registry_path}")
        except (IOError, json.JSONEncodeError) as e:
            print(f"Warning: Failed to create registry file: {e}")
    
    return registry_path

def load_vector_store_registry() -> Dict[str, Any]:
    """
    Load the vector store registry from disk.
    
    Returns:
        Dict: The registry data, or empty dict if file doesn't exist.
    """
    registry_path = get_vector_store_registry_path()
    
    if not registry_path.exists():
        return {"stores": {}}
    
    try:
        with open(registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load registry file: {e}")
        return {"stores": {}}

def save_vector_store_registry(registry: Dict[str, Any]) -> bool:
    """
    Save the vector store registry to disk.
    
    Args:
        registry: The registry data to save.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Ensure app directory exists
        get_or_create_app_dir()
        
        registry_path = get_vector_store_registry_path()
        
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        return True
    except (IOError, json.JSONEncodeError) as e:
        print(f"Error: Failed to save registry file: {e}")
        return False

def add_vector_store_to_registry(
    name: str,
    collection_name: str,
    persist_directory: str,
    description: str = "",
    debug: bool = False
) -> bool:
    """
    Add a new vector store to the registry.
    
    Args:
        name: Human-readable name for the store (used as key)
        collection_name: Name of the collection in the vector database
        persist_directory: Directory where the vector store is persisted
        description: Optional description
        debug: Whether to show debug output
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if debug:
        print(f"Adding vector store '{name}' to registry...")
    
    registry = load_vector_store_registry()
    if debug:
        print(f"Current registry has {len(registry.get('stores', {}))} stores")
    
    store_entry = {
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "description": description
    }
    
    registry["stores"][name] = store_entry
    success = save_vector_store_registry(registry)
    
    if success:
        if debug:
            print(f"Successfully added '{name}' to registry")
            # Verify by reloading
            updated_registry = load_vector_store_registry()
            print(f"Updated registry now has {len(updated_registry.get('stores', {}))} stores")
    else:
        if debug:
            print(f"Failed to save '{name}' to registry")
    
    return success

def list_vector_stores() -> Dict[str, Dict[str, Any]]:
    """
    List all vector stores in the registry.
    
    Returns:
        Dict: Dictionary with store names as keys and store data as values.
    """
    registry = load_vector_store_registry()
    return registry.get("stores", {})

def format_docs(docs: list[Document]) -> str:
    """
    Combine the page_content of retrieved documents into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)
