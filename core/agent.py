"""
chains.py
#########

Simple agentic chat implementation with tools and memory.
"""

# Imports
from langchain.agents import AgentExecutor, create_tool_calling_agent # remove when using langgraph only

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from typing import List, Any, Dict, Optional, Annotated, TypedDict
from pydantic import BaseModel, Field

from core.models import chat_model
from core.prompts import AGENT_SYSTEM_PROMPT

# --- Single State Definition using TypedDict ---
class AgentState(TypedDict):
    """
    The state of the agent.

    Attributes:
        messages: The list of messages that have been exchanged.
                  This is the primary way that memory is stored.
                  The 'add_messages' reducer is used to append new messages.
        input: The current user input. This is cleared after being processed.
        topic: The topic/subject matter for the agent's context.
        debug: A flag to enable or disable debug printing.
        loop_count: The number of loops in the current turn.
        max_loops: The maximum number of loops allowed per turn.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    input: str
    topic: str
    debug: bool
    loop_count: int
    max_loops: int
    output: str


def create_agent_with_tools_and_memory(tools: List[object], topic: str, debug: bool = False, max_loops: int = 6):
    """
    Create an agent with tools and conversational memory.
    
    Args:
        tools: List of LangChain tools the agent can use
        topic: The topic/subject matter for the agent's context
        debug: Whether to show verbose debug output (retrieved documents, tool calls, etc.)
        max_loops: Maximum number of loops allowed per question (default: 3)
        
    Returns:
        A runnable agent with message history that can be invoked with:
        agent.invoke({"input": "message"}, config={"configurable": {"session_id": "session_id"}})
    """
    memory = MemorySaver()
    tool_node = ToolNode(tools)
    model_with_tools = chat_model.bind_tools(tools)

    def call_model(state: AgentState) -> dict:
        """
        Node for calling the LLM.
        """
        if debug:
            print(f"\n--- Calling Model (Loop: {state['loop_count']}) ---")
            print(f"[DEBUG] call_model - Input: {state['input']}")
            print(f"[DEBUG] call_model - Current messages count: {len(state['messages'])}")
        
        # We will now use the built-in message adder, so we just need to prepare the new messages
        new_messages = []
        
        # Add system message only if no messages exist yet
        if not state['messages']:
            new_messages.append(SystemMessage(content=f"You are working on: {topic}. {AGENT_SYSTEM_PROMPT}"))
        
        # Add the current user input as a HumanMessage
        if state['input'] and state['loop_count'] == 0:
            if debug:
                print(f"[DEBUG] call_model - Adding user input as HumanMessage")
            new_messages.append(HumanMessage(content=state['input']))

        # The full history is passed to the model
        # Note: state['messages'] is now automatically loaded by the checkpointer!
        messages_to_send = state['messages'] + new_messages

        if debug:
            print(f"[DEBUG] call_model - Total messages to send: {len(messages_to_send)}")

        response = model_with_tools.invoke(messages_to_send)
        new_messages.append(response)

        return {
            "messages": new_messages,  # Return ONLY the new messages to be appended
            "input": "", # Clear the input field after processing
            "loop_count": state['loop_count'] + 1,
            "max_loops": max_loops,
            "output": response.content if isinstance(response, AIMessage) else ""
        }

    def call_tools(state: AgentState) -> dict:
        """
        Node for calling tools with the current state.
        """
        last_message = state['messages'][-1]

        if debug:
            print(f"[DEBUG] call_tools - Last message type: {type(last_message)}")
            print(f"[DEBUG] call_tools - Tool calls: {getattr(last_message, 'tool_calls', 'None')}")
            print(f"[DEBUG] call_tools - Loop count: {state['loop_count']}")

        tool_results = tool_node.invoke({"messages": [last_message]})

        if debug:
            print(f"[DEBUG] call_tools - Tool results: {tool_results}")
        
        return {
            "messages": tool_results["messages"],
            "loop_count": state['loop_count'] + 1
        }
    
    def should_continue(state: AgentState) -> str:
        """
        Check if the agent should continue processing.
        """
        last_message = state['messages'][-1]
        tool_calls = getattr(last_message, 'tool_calls', [])

        if debug:
            print(f"[DEBUG] should_continue - Last message type: {type(last_message)}")
            print(f"[DEBUG] should_continue - Loop count: {state['loop_count']}/{state['max_loops']}")
            print(f"[DEBUG] should_continue - Tool calls count: {len(tool_calls)}")
            if tool_calls:
                print(f"[DEBUG] should_continue - Tool calls: {tool_calls}")
        
        # Check if we've exceeded the maximum number of loops
        if state['loop_count'] >= state['max_loops']:
            if debug:
                print(f"[DEBUG] should_continue - Maximum loops reached, ending")
            return "end"
        
        # Simply check if tool_calls list has any items
        if tool_calls:
            return "continue"
        else:
            return "end"
    
    # Create the graph
    graph_builder = StateGraph(AgentState)
    # Add nodes
    graph_builder.add_node("call_model", call_model)
    graph_builder.add_node("call_tools", call_tools)
    # Define flow
    graph_builder.set_entry_point("call_model")
    graph_builder.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "continue": "call_tools",
            "end": END
        }
    )
    graph_builder.add_edge("call_tools", "call_model")
    # Compile the graph with memory
    return graph_builder.compile(checkpointer=memory)


def old_create_agent_with_tools_and_memory(tools: List[object], topic: str, debug: bool = False):
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
