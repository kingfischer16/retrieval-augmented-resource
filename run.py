"""
run.py
######

This is the main entry point for the application. Running this script in a terminal will provide the user with
a CLI, a list of available vector stores currently in local app data cache, and the following options:
 1. Create a new vector store: This will prompt the user to enter a file or folder path, as well as a name for the vector store.
 2. Chat with existing vector store: This will create a new chat session and connect to the selected vector store.
 3. Exit: This will exit the application.

Creating a new vector store will end with the user being returned to the main menu CLI with the options above, with the
newly created vector store now available in the list of vector stores.

Chatting with an existing vector store will put the using in a command line chat session, where the user can ask questions
as though the chat agent was an expert in the field of the vector store's data. The user can exit the chat session
by typing `:exit`, which will return them to the main menu CLI.

"""
