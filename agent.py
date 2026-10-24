"""NVIDIA Nemotron Chat Agent using LangChain & LangGraph."""

import os
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

import config
from tools import agent_tools

SYSTEM_PROMPT = """You are an intelligent, capable, and helpful AI assistant powered by NVIDIA's Nemotron 3 Ultra reasoning model.
You excel at multi-step reasoning, logical problem-solving, structured code generation, and deep analysis.

Guidelines:
1. Always maintain a friendly, professional, and clear tone.
2. Use your available tools when necessary:
   - For real-time information or events: use `search_web`.
   - For mathematical calculations: use `calculate` to ensure exact accuracy.
   - For current time/date questions: use `get_current_datetime`.
   - For taking notes, jotting down ideas, or saving memos: use `create_note`.
   - For reviewing or searching past notes: use `list_notes`.
3. If a question can be answered accurately without tools, answer directly.
4. When performing calculations or research, reason step-by-step before delivering the final answer.
5. Format code blocks with appropriate language tags and maintain clean Markdown formatting.
"""

def get_llm():
    """Initializes the NVIDIA Nemotron LLM using ChatNVIDIA or ChatOpenAI fallback."""
    # Priority 1: Official langchain_nvidia_ai_endpoints
    try:
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        return ChatNVIDIA(
            model=config.NVIDIA_MODEL,
            api_key=config.NVIDIA_API_KEY,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            timeout=120,
        )
    except (ImportError, Exception):
        pass

    # Priority 2: OpenAI-compatible NVIDIA NIM endpoint
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.NVIDIA_MODEL,
            api_key=config.NVIDIA_API_KEY,
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            timeout=120,
        )
    except ImportError as e:
        raise ImportError(
            "Please install either 'langchain-nvidia-ai-endpoints' or 'langchain-openai'.\n"
            "Run: pip install langchain-nvidia-ai-endpoints"
        ) from e


def create_agent(tools: Optional[List] = None, checkpointer: Optional[MemorySaver] = None):
    """Creates a ReAct agent powered by NVIDIA Nemotron with session memory."""
    if tools is None:
        tools = agent_tools
        
    if checkpointer is None:
        checkpointer = MemorySaver()

    llm = get_llm()

    # Create the agent using LangGraph's prebuilt ReAct agent
    # Compatible with LangGraph 0.2+ prompt / state_modifier
    try:
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )
    except TypeError:
        # Backward compatibility for older versions using state_modifier
        agent = create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )

    return agent


def create_rag_agent(checkpointer: Optional[MemorySaver] = None):
    """Creates an Agentic RAG graph workflow powered by LangGraph and NVIDIA Nemotron."""
    from graph import create_rag_graph
    return create_rag_graph(checkpointer=checkpointer)


def create_adaptive_agent(checkpointer: Optional[MemorySaver] = None):
    """Creates a Unified Adaptive Agent that autonomously routes queries between RAG and ReAct tools."""
    from graph import create_unified_agent_graph
    return create_unified_agent_graph(checkpointer=checkpointer)


