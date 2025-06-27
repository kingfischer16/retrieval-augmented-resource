"""
chains.py
#########

Simple agentic chat implementation with tools and memory.
"""

from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from core.models import chat_model
from core.prompts import AGENT_SYSTEM_PROMPT


def create_agent_with_tools_and_memory(tools: List[object], topic: str, debug: bool = False) -> RunnableWithMessageHistory:
    """
    Create an agent with tools and conversational memory.
    
    Args:
        tools: List of LangChain tools the agent can use
        topic: The topic/subject matter for the agent's context
        debug: Whether to show verbose debug output (retrieved documents, tool calls, etc.)
        
    Returns:
        A runnable agent with message history that can be invoked with:
        agent.invoke({"input": "message"}, config={"configurable": {"session_id": "session_id"}})
    """
    
    # Create the agent prompt
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are working on the topic: {topic}. {AGENT_SYSTEM_PROMPT}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_tool_calling_agent(
        llm=chat_model,
        tools=tools,
        prompt=agent_prompt
    )
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=debug,  # Only show verbose output in debug mode
        return_intermediate_steps=debug,  # Only return intermediate steps in debug mode
        handle_parsing_errors=True
    )
    
    # Store for session message histories
    session_store: Dict[str, ChatMessageHistory] = {}
    
    def get_session_history(session_id: str) -> ChatMessageHistory:
        """Get or create a chat message history for a session."""
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]
    
    # Create the agent with message history
    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return agent_with_history
