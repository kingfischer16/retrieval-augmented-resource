"""
run.py
######

This is the main entry point for the application. Running this script in a terminal will provide the user with
a CLI, a list of available vector stores currently in local app data cache, and the following options:
 1. Create a new vector store: This will prompt the user to enter a file or folder path, as well as a name for the vector store.
 2. Chat with existing vector store: This will create a new chat session and connec            choice = get_user_input("Select an option (1-5): ", Colors.YELLOW)
            
            if choice == '1':
                create_new_vector_store()
            elif choice == '2':
                chat_with_vector_store()
            elif choice == '3':
                delete_vector_store()
            elif choice == '4':
                toggle_debug_mode()
            elif choice == '5':
                print(f"\n{Colors.CYAN}Thank you for using the Retrieval Augmented Resource Chat System!{Colors.END}")
                print(f"{Colors.CYAN}Goodbye!{Colors.END}\n")
                sys.exit(0)
            else:
                print_error("Invalid option. Please select 1, 2, 3, 4, or 5.")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")ed vector store.
 3. Exit: This will exit the application.

Creating a new vector store will end with the user being returned to the main menu CLI with the options above, with the
newly created vector store now available in the list of vector stores.

Chatting with an existing vector store will put the using in a command line chat session, where the user can ask questions
as though the chat agent was an expert in the field of the vector store's data. The user can exit the chat session
by typing `:exit`, which will return them to the main menu CLI.

"""

import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Any

# Color codes for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Import our core functionality
from core.utils import list_vector_stores, add_vector_store_to_registry, remove_vector_store_from_registry
from core.vector_store import create_vector_store, create_retriever_tool_from_vector_store
from core.agent import create_agent_with_tools_and_memory

# Global debug flag
DEBUG_MODE = False


def print_header():
    """Print the application header."""
    debug_status = f" {Colors.YELLOW}[DEBUG MODE ON]{Colors.END}" if DEBUG_MODE else ""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print("  RETRIEVAL AUGMENTED RESOURCE CHAT SYSTEM")
    print(f"{'='*60}{Colors.END}{debug_status}\n")


def print_main_menu():
    """Print the main menu options."""
    print(f"{Colors.BOLD}{Colors.WHITE}Main Menu:{Colors.END}")
    print(f"{Colors.YELLOW}1.{Colors.END} Create a new vector store")
    print(f"{Colors.YELLOW}2.{Colors.END} Chat with existing vector store")
    print(f"{Colors.YELLOW}3.{Colors.END} Delete an existing vector store")
    print(f"{Colors.YELLOW}4.{Colors.END} Toggle debug mode {'(ON)' if DEBUG_MODE else '(OFF)'}")
    print(f"{Colors.YELLOW}5.{Colors.END} Exit")
    print()


def get_user_input(prompt: str, color: str = Colors.WHITE) -> str:
    """Get user input with color formatting."""
    return input(f"{color}{prompt}{Colors.END}")


def print_error(message: str):
    """Print error message in red."""
    print(f"{Colors.RED}Error: {message}{Colors.END}")


def print_success(message: str):
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_info(message: str):
    """Print info message in cyan."""
    print(f"{Colors.CYAN}ℹ {message}{Colors.END}")


def list_available_vector_stores() -> Dict[str, Dict[str, Any]]:
    """List available vector stores and return the registry."""
    stores = list_vector_stores()
    
    if not stores:
        print_info("No vector stores found. Create one first using option 1.")
        return {}
    
    print(f"\n{Colors.BOLD}Available Vector Stores:{Colors.END}")
    print(f"{Colors.YELLOW}{'No.':<4} {'Name':<30} {'Description':<40}{Colors.END}")
    print("-" * 76)
    
    store_list = list(stores.keys())
    for i, store_name in enumerate(store_list, 1):
        store_info = stores[store_name]
        description = store_info.get('description', 'No description')
        print(f"{Colors.WHITE}{i:<4} {store_name:<30} {description[:37] + '...' if len(description) > 40 else description:<40}{Colors.END}")
    
    print()
    return stores


def create_new_vector_store():
    """Handle vector store creation workflow."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Create New Vector Store{Colors.END}")
    print("-" * 30)
    
    # Get file or folder path
    while True:
        path = get_user_input("Enter file or folder path: ", Colors.YELLOW).strip()
        if not path:
            print_error("Path cannot be empty.")
            continue
        
        path_obj = Path(path)
        if not path_obj.exists():
            print_error(f"Path '{path}' does not exist.")
            continue
        
        if not (path_obj.is_file() or path_obj.is_dir()):
            print_error(f"Path '{path}' is not a valid file or directory.")
            continue
        
        break
    
    # Get store name
    while True:
        store_name = get_user_input("Enter a name for the vector store: ", Colors.YELLOW).strip()
        if not store_name:
            print_error("Store name cannot be empty.")
            continue
        
        # Check if name already exists
        existing_stores = list_vector_stores()
        if store_name in existing_stores:
            print_error(f"A vector store named '{store_name}' already exists.")
            continue
        
        break
    
    # Get description (optional)
    description = get_user_input("Enter a description (optional, press Enter to skip): ", Colors.YELLOW).strip()
    if not description:
        description = f"Vector store created from {path_obj.name}"
    
    # Generate unique collection name
    collection_name = f"collection_{uuid.uuid4().hex[:8]}"
    
    print(f"\n{Colors.CYAN}Creating vector store...{Colors.END}")
    print(f"Path: {path}")
    print(f"Name: {store_name}")
    print(f"Description: {description}")
    print()
    
    # Check if it's a directory with many files
    if path_obj.is_dir():
        try:
            # Count files in directory
            file_count = sum(1 for file in path_obj.rglob('*') if file.is_file())
            print(f"Directory contains {file_count} files.")
            
            if file_count > 500:
                print(f"{Colors.YELLOW}Warning: Processing {file_count} files may take a very long time and use significant API quota.{Colors.END}")
                print(f"{Colors.YELLOW}Consider processing a smaller subset first.{Colors.END}")
                
                # Offer option to limit the number of files
                limit_files = get_user_input("Do you want to limit the number of files? (y/N): ", Colors.YELLOW).strip().lower()
                if limit_files in ['y', 'yes']:
                    while True:
                        try:
                            max_files = int(get_user_input("Enter maximum number of files to process: ", Colors.YELLOW).strip())
                            if max_files > 0:
                                break
                            else:
                                print_error("Please enter a positive number.")
                        except ValueError:
                            print_error("Please enter a valid number.")
                    
                    print(f"Will process only the first {max_files} files found.")
                    # We'll pass this limit to the vector store creation function
                else:
                    confirm = get_user_input("Do you want to continue with all files? (y/N): ", Colors.YELLOW).strip().lower()
                    if confirm not in ['y', 'yes']:
                        print("Vector store creation cancelled.")
                        input(f"\n{Colors.YELLOW}Press Enter to return to main menu...{Colors.END}")
                        return
                    max_files = None
            else:
                max_files = None
        except Exception:
            max_files = None  # If we can't count files, just proceed
    else:
        max_files = None
    
    try:
        # Create the vector store (this function handles registration internally)
        success = create_vector_store(
            file_or_folder_path=path,
            store_name=store_name,
            description=description,
            collection_name=collection_name,
            text_splitter="recursive",  # Default to recursive splitter
            debug=DEBUG_MODE,
            max_files=max_files if 'max_files' in locals() else None
        )
        
        if success:
            print_success(f"Vector store '{store_name}' created successfully!")
        else:
            print_error(f"Failed to create vector store '{store_name}'. See error messages above.")
        
    except KeyboardInterrupt:
        print_error("Vector store creation was interrupted by user.")
    except Exception as e:
        print_error(f"Failed to create vector store: {str(e)}")
        if DEBUG_MODE:
            import traceback
            print(f"\n{Colors.RED}Full error traceback:{Colors.END}")
            traceback.print_exc()
    
    input(f"\n{Colors.YELLOW}Press Enter to return to main menu...{Colors.END}")


def chat_with_vector_store():
    """Handle chat workflow with vector store selection."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Chat with Vector Store{Colors.END}")
    print("-" * 30)
    
    # List available stores
    stores = list_available_vector_stores()
    if not stores:
        input(f"\n{Colors.YELLOW}Press Enter to return to main menu...{Colors.END}")
        return
    
    # Get user selection
    store_list = list(stores.keys())
    while True:
        try:
            choice = get_user_input(f"Select a vector store (1-{len(store_list)}): ", Colors.YELLOW)
            choice_num = int(choice)
            if 1 <= choice_num <= len(store_list):
                selected_store_name = store_list[choice_num - 1]
                break
            else:
                print_error(f"Please enter a number between 1 and {len(store_list)}.")
        except ValueError:
            print_error("Please enter a valid number.")
    
    # Get selected store info
    store_info = stores[selected_store_name]
    collection_name = store_info['collection_name']
    persist_directory = store_info['persist_directory']
    description = store_info.get('description', '')
    
    print(f"\n{Colors.GREEN}Connecting to '{selected_store_name}'...{Colors.END}")
    
    try:
        # Create retriever tool
        retriever_tool = create_retriever_tool_from_vector_store(
            collection_name=collection_name,
            persist_directory=persist_directory,
            store_name=selected_store_name
        )
        
        # Create topic string
        topic = f"{selected_store_name}"
        if description:
            topic += f": {description}"
        
        # Create agent with tools and memory
        agent = create_agent_with_tools_and_memory(
            tools=[retriever_tool],
            topic=topic,
            debug=DEBUG_MODE  # Pass the global debug flag
        )
        
        # Generate session ID
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        print_success(f"Connected to '{selected_store_name}'. Chat session started!")
        print(f"{Colors.YELLOW}Type ':exit' to return to main menu.{Colors.END}\n")
        
        # Start chat loop
        start_chat_session(agent, session_id, selected_store_name, topic)
        
    except Exception as e:
        print_error(f"Failed to connect to vector store: {str(e)}")
        input(f"\n{Colors.YELLOW}Press Enter to return to main menu...{Colors.END}")


def delete_vector_store():
    """Handle vector store deletion workflow."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Delete Vector Store{Colors.END}")
    print("-" * 30)
    
    # List available stores
    stores = list_available_vector_stores()
    if not stores:
        input(f"\n{Colors.YELLOW}Press Enter to return to main menu...{Colors.END}")
        return
    
    # Get user selection
    store_list = list(stores.keys())
    while True:
        try:
            choice = get_user_input(f"Select a vector store to delete (1-{len(store_list)}): ", Colors.YELLOW)
            choice_num = int(choice)
            if 1 <= choice_num <= len(store_list):
                selected_store_name = store_list[choice_num - 1]
                break
            else:
                print_error(f"Please enter a number between 1 and {len(store_list)}.")
        except ValueError:
            print_error("Please enter a valid number.")
    
    # Get selected store info
    store_info = stores[selected_store_name]
    persist_directory = store_info['persist_directory']
    description = store_info.get('description', 'No description')
    
    # Show details and confirm deletion
    print(f"\n{Colors.RED}{Colors.BOLD}WARNING: This action cannot be undone!{Colors.END}")
    print(f"\n{Colors.YELLOW}Vector store details:{Colors.END}")
    print(f"  Name: {selected_store_name}")
    print(f"  Description: {description}")
    print(f"  Persist Directory: {persist_directory}")
    
    # Double confirmation
    print(f"\n{Colors.RED}This will permanently delete:{Colors.END}")
    print(f"  • The vector store entry from the registry")
    print(f"  • All persisted vector data files")
    print(f"  • Any embeddings and indexes")
    
    confirm1 = get_user_input(f"\nAre you sure you want to delete '{selected_store_name}'? (type 'yes' to confirm): ", Colors.RED).strip()
    
    if confirm1.lower() != 'yes':
        print(f"\n{Colors.CYAN}Deletion cancelled.{Colors.END}")
        input(f"\n{Colors.YELLOW}Press Enter to return to main menu...{Colors.END}")
        return
    
    # Final confirmation
    confirm2 = get_user_input(f"Type the store name '{selected_store_name}' to confirm deletion: ", Colors.RED).strip()
    
    if confirm2 != selected_store_name:
        print(f"\n{Colors.CYAN}Deletion cancelled - store name doesn't match.{Colors.END}")
        input(f"\n{Colors.YELLOW}Press Enter to return to main menu...{Colors.END}")
        return
    
    # Proceed with deletion
    print(f"\n{Colors.CYAN}Deleting vector store '{selected_store_name}'...{Colors.END}")
    
    try:
        success = remove_vector_store_from_registry(selected_store_name, debug=DEBUG_MODE)
        
        if success:
            print_success(f"Vector store '{selected_store_name}' has been permanently deleted!")
            print(f"{Colors.CYAN}• Removed from registry{Colors.END}")
            print(f"{Colors.CYAN}• Deleted all persisted files{Colors.END}")
        else:
            print_error(f"Failed to delete vector store '{selected_store_name}'. It may not exist in the registry.")
            
    except Exception as e:
        print_error(f"An error occurred while deleting the vector store: {str(e)}")
        if DEBUG_MODE:
            import traceback
            print(f"\n{Colors.RED}Full error traceback:{Colors.END}")
            traceback.print_exc()
    
    input(f"\n{Colors.YELLOW}Press Enter to return to main menu...{Colors.END}")


def start_chat_session(agent, session_id: str, store_name: str, topic: str):
    """Handle the chat session loop."""
    print(f"{Colors.BOLD}{Colors.CYAN}=== Chat Session with '{store_name}' ==={Colors.END}")
    print(f"{Colors.GREEN}[CHAT]{Colors.END} Hello! I'm ready to help you with questions about {store_name}. What would you like to know?")
    
    while True:
        # Get user input
        try:
            user_input = get_user_input(f"\n{Colors.BLUE}[USER]{Colors.END} ", Colors.BLUE).strip()
            
            # Check for exit command
            if user_input.lower() == ':exit':
                print(f"\n{Colors.GREEN}[CHAT]{Colors.END} Goodbye! Returning to main menu.")
                break
            
            if not user_input:
                continue
            
            # Get agent response
            print(f"{Colors.CYAN}Thinking...{Colors.END}", end="", flush=True)
            
            try:
                config = {"configurable": {"thread_id": session_id}}
                response = agent.invoke(
                    {
                        "input": user_input,
                        "topic": topic,  # ← Add missing comma
                        "debug": DEBUG_MODE,
                        "loop_count": 0,  # ← Initialize for new questions
                        "max_loops": 3    # ← Initialize max_loops
                    },
                    config=config
                )
                
                # Clear the "Thinking..." message and print response
                print(f"\r{' ' * 12}\r", end="")  # Clear the line
                print("Printing output...")
                print(f"{Colors.GREEN}[CHAT]{Colors.END} {response['output']}")
                
            except Exception as e:
                print(f"\r{' ' * 12}\r", end="")  # Clear the line
                print(f"{Colors.RED}[CHAT]{Colors.END} I apologize, but I encountered an error: {str(e)}")
                
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Chat session interrupted. Returning to main menu.{Colors.END}")
            break
        except EOFError:
            print(f"\n\n{Colors.YELLOW}Chat session ended. Returning to main menu.{Colors.END}")
            break
    
    input(f"\n{Colors.YELLOW}Press Enter to return to main menu...{Colors.END}")


def toggle_debug_mode():
    """Toggle the global debug mode on/off."""
    global DEBUG_MODE
    DEBUG_MODE = not DEBUG_MODE
    
    status = "ENABLED" if DEBUG_MODE else "DISABLED"
    color = Colors.GREEN if DEBUG_MODE else Colors.RED
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Debug Mode Toggle{Colors.END}")
    print("-" * 20)
    print(f"Debug mode is now {color}{status}{Colors.END}")
    
    if DEBUG_MODE:
        print(f"\n{Colors.YELLOW}Debug mode will show:{Colors.END}")
        print(f"  • Retrieved documents from vector store")
        print(f"  • Tool call details")
        print(f"  • Verbose agent execution logs")
    else:
        print(f"\n{Colors.YELLOW}Debug mode is off - only chat messages will be shown{Colors.END}")
    
    input(f"\n{Colors.YELLOW}Press Enter to return to main menu...{Colors.END}")


def main():
    """Main application loop."""
    try:
        while True:
            # Clear screen (works on both Windows and Unix) - but not in debug mode
            if not DEBUG_MODE:
                os.system('cls' if os.name == 'nt' else 'clear')
            
            print_header()
            print_main_menu()
            
            choice = get_user_input("Select an option (1-5): ", Colors.YELLOW)
            
            if choice == '1':
                create_new_vector_store()
            elif choice == '2':
                chat_with_vector_store()
            elif choice == '3':
                delete_vector_store()
            elif choice == '4':
                toggle_debug_mode()
            elif choice == '5':
                print(f"\n{Colors.CYAN}Thank you for using the Retrieval Augmented Resource Chat System!{Colors.END}")
                print(f"{Colors.CYAN}Goodbye!{Colors.END}\n")
                sys.exit(0)
            else:
                print_error("Invalid option. Please select 1, 2, 3, 4, or 5.")
                input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.CYAN}Application interrupted. Goodbye!{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print_error(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
