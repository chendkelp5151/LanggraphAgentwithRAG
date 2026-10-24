"""State definition for Adaptive Unified Agent in LangGraph."""

from typing import List, TypedDict, Annotated, Optional
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class UnifiedAgentState(TypedDict):
    """Represents the shared state of our Unified Adaptive Router Graph.
    
    Attributes:
        question: User's query.
        generation: LLM generated answer.
        documents: List of retrieved (and graded) documents.
        web_search: Whether to execute web search fallback.
        iterations: Number of query transformation iterations executed.
        route: The selected execution pathway ('rag' or 'react').
        messages: Conversation message history with add_messages reducer.
    """
    question: str
    generation: str
    documents: List[Document]
    web_search: bool
    iterations: int
    route: str
    messages: Annotated[List[BaseMessage], add_messages]


# Backward compatibility alias
RAGState = UnifiedAgentState
