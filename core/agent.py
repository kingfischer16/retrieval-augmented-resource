"""
chains.py
#########

Simple agentic chat implementation with tools and memory.
"""

# Imports
from langchain.agents import AgentExecutor, create_tool_calling_agent # remove when using langgraph only

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.models import chat_model
from core.prompts import AGENT_SYSTEM_PROMPT

class AgentState(BaseModel):
    """State for the LangGraph agent."""
    messages: List[BaseMessage] = Field(default_factory=list, description="Chat history messages")
    input: str = Field(default="", description="Current user input")
    intermediate_steps: List[Dict[str, Any]] = Field(default_factory=list, description="Tool call steps")
    output: Optional[str] = Field(default=None, description="Final agent output")
    session_id: str = Field(default="", description="Session identifier")
    topic: str = Field(default="", description="Topic/subject matter")
    debug: bool = Field(default=False, description="Debug mode flag")
    loop_count: int = Field(default=0, description="Number of loops in current question")
    max_loops: int = Field(default=10, description="Maximum number of loops allowed")


def create_agent_with_tools_and_memory(tools: List[object], topic: str, debug: bool = False, max_loops: int = 3):
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
    tool_node = ToolNode(tools)

    def call_model(state: AgentState) -> dict:
        """
        Node for calling the LLM.
        """
        if debug:
            print(f"[DEBUG] call_model - Input: {state.input}")
            print(f"[DEBUG] call_model - Current messages count: {len(state.messages)}")
        
        # Build the message sequence properly
        messages = []
        
        # Add system message only if no messages exist yet
        if not state.messages:
            messages.append(SystemMessage(content=f"You are working on: {state.topic}. {AGENT_SYSTEM_PROMPT}"))
        
        # Add existing chat history
        messages.extend(state.messages)
        
        # Add current user input
        if state.input:
            messages.append(HumanMessage(content=state.input))

        if debug:
            print(f"[DEBUG] call_model - Total messages to send: {len(messages)}")

        model_with_tools = chat_model.bind_tools(tools)
        response = model_with_tools.invoke(messages)

        if debug:
            print(f"[DEBUG] call_model - Response type: {type(response)}")
            print(f"[DEBUG] call_model - Response content: {response.content}")
            print(f"[DEBUG] call_model - Has tool_calls attr: {hasattr(response, 'tool_calls')}")
            if hasattr(response, 'tool_calls'):
                print(f"[DEBUG] call_model - Tool calls: {response.tool_calls}")
        
        messages.append(response)  # Use the actual response with tool calls
        
        return {
            "messages": messages,
            "output": response.content if hasattr(response, 'content') else str(response),
            "loop_count": 0,  # Reset loop count for new question
            "max_loops": max_loops  # Set max loops
        }

    def call_tools(state: AgentState) -> dict:
        """
        Node for calling tools with the current state.
        """
        last_message = state.messages[-1]

        if debug:
            print(f"[DEBUG] call_tools - Last message type: {type(last_message)}")
            print(f"[DEBUG] call_tools - Tool calls: {getattr(last_message, 'tool_calls', 'None')}")
            print(f"[DEBUG] call_tools - Loop count: {state.loop_count}")

        tool_results = tool_node.invoke({"messages": [last_message]})

        if debug:
            print(f"[DEBUG] call_tools - Tool results: {tool_results}")
        
        return {
            "messages": state.messages + tool_results["messages"],
            "loop_count": state.loop_count + 1
        }
    
    def should_continue(state: AgentState) -> str:
        """
        Check if the agent should continue processing.
        """
        last_message = state.messages[-1]

        if debug:
            print(f"[DEBUG] should_continue - Last message type: {type(last_message)}")
            print(f"[DEBUG] should_continue - Has tool_calls: {hasattr(last_message, 'tool_calls')}")
            print(f"[DEBUG] should_continue - Loop count: {state.loop_count}/{state.max_loops}")
            if hasattr(last_message, 'tool_calls'):
                print(f"[DEBUG] should_continue - Tool calls: {last_message.tool_calls}")
        
        # Check if we've exceeded the maximum number of loops
        if state.loop_count >= state.max_loops:
            if debug:
                print(f"[DEBUG] should_continue - Maximum loops reached ({state.max_loops}), ending")
            return "end"
        
        # Check if the last message has tool calls
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        else:
            return "end"  # Return string, not END constant
    
    # Create the graph
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("call_model", call_model)
    graph_builder.add_node("call_tools", call_tools)
    graph_builder.set_entry_point("call_model")
    graph_builder.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "continue": "call_tools",
            "end": END  # Map string to END constant
        }
    )
    graph_builder.add_edge("call_tools", "call_model")
    return graph_builder.compile()


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
