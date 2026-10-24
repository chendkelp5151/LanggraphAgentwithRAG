"""LangGraph Adaptive Router Graph: Dynamically routes between RAG and ReAct Tools based on query intent."""

import os
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from rag.state import UnifiedAgentState
from rag.ingest import get_vectorstore
from agent import get_llm, SYSTEM_PROMPT
from tools import agent_tools, search_web


# ---------------------------------------------------------
# Node 0: Query Router (LLM Intent Classifier)
# ---------------------------------------------------------
def router_node(state: UnifiedAgentState) -> Dict[str, Any]:
    """Analyzes the user's query and dynamically chooses between RAG and ReAct pathways."""
    question = state.get("question", "")
    
    # Extract query from messages if question not explicitly set
    if not question and state.get("messages"):
        last_msg = state["messages"][-1]
        question = last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg.content)

    print(f"\n[ROUTER] Analyzing query intent for: '{question}'")
    
    router_prompt = (
        "You are an expert AI query router.\n"
        "Analyze the following user input and decide the best execution pathway:\n\n"
        "1. 'rag': Questions asking about internal documents, company knowledge, uploaded guides, "
        "specifications (like Nemotron specs, technical architectures), or PDF/document QA.\n"
        "2. 'react': Questions requiring mathematical calculations, current date/time, personal notes, "
        "live general web search, general chatting, or conversational requests.\n\n"
        f"User Query: {question}\n\n"
        "Return ONLY the word 'rag' or 'react'. No explanation."
    )

    llm = get_llm()
    try:
        response = llm.invoke([HumanMessage(content=router_prompt)])
        decision = response.content.strip().lower()
        route = "rag" if "rag" in decision else "react"
    except Exception as e:
        print(f"[ROUTER ERROR] Router failed: {e}. Defaulting to 'react'")
        route = "react"

    print(f"[ROUTER DECISION] Route chosen: '{route.upper()}'")
    return {"route": route, "question": question}


def route_decision(state: UnifiedAgentState) -> Literal["retrieve", "react_agent"]:
    """Conditional edge directing flow based on router determination."""
    if state.get("route") == "rag":
        return "retrieve"
    return "react_agent"


# ---------------------------------------------------------
# ReAct Pathway: Agent Node & Tool Calling
# ---------------------------------------------------------
def react_agent_node(state: UnifiedAgentState) -> Dict[str, Any]:
    """ReAct agent node with all tools bound (calculator, notes, time, web search)."""
    messages = state.get("messages", [])
    print("\n[REACT NODE] Invoking tool-calling model...")
    
    llm = get_llm()
    llm_with_tools = llm.bind_tools(agent_tools)
    
    # Inject system prompt if not present
    full_messages = messages
    if not any(isinstance(m, SystemMessage) for m in messages):
        full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm_with_tools.invoke(full_messages)
    return {
        "messages": [response],
        "generation": response.content if isinstance(response.content, str) else str(response.content)
    }


# ---------------------------------------------------------
# RAG Pathway: Nodes (Retrieve, Grade, Transform, Fallback, Generate)
# ---------------------------------------------------------
def retrieve(state: UnifiedAgentState) -> Dict[str, Any]:
    """Retrieves top-k documents from ChromaDB matching the question with page-aware filtering."""
    question = state["question"]
    print(f"\n[RAG: RETRIEVE] Querying ChromaDB for: '{question}'")
    
    docs: List[Document] = []
    try:
        vectorstore = get_vectorstore()
        
        # 1. Page-aware retrieval: Check if query asks for a specific page (e.g. 'page 1', 'page 3')
        import re
        page_match = re.search(r'\bpage\s*(\d+)\b', question, re.IGNORECASE)
        if page_match:
            target_page = int(page_match.group(1))
            # PyPDFLoader indexes pages 0-indexed (page 1 is 0)
            page_filter = {"page": target_page - 1}
            try:
                page_results = vectorstore.get(where=page_filter)
                if page_results and page_results.get("documents"):
                    for text, meta in zip(page_results["documents"], page_results["metadatas"]):
                        docs.append(Document(page_content=text, metadata=meta))
                    print(f"[RAG: RETRIEVE] Page-aware filter matched {len(docs)} chunks for Page {target_page}")
            except Exception as e:
                print(f"[RAG: RETRIEVE] Page filter lookup failed: {e}")

        # 2. Dense Semantic Vector Search (k=6)
        sim_docs = vectorstore.similarity_search(question, k=6)
        
        # Merge avoiding duplicates
        seen_texts = {d.page_content for d in docs}
        for d in sim_docs:
            if d.page_content not in seen_texts:
                docs.append(d)
                seen_texts.add(d.page_content)

        print(f"[RAG: RETRIEVE] Total chunks retrieved for evaluation: {len(docs)}")
    except Exception as e:
        print(f"[RAG: RETRIEVE ERROR] {e}")

    return {"documents": docs, "question": question}


def grade_documents(state: UnifiedAgentState) -> Dict[str, Any]:
    """Determines whether the retrieved documents are relevant to the question."""
    question = state["question"]
    documents = state.get("documents", [])
    
    print("\n[RAG: GRADE_DOCUMENTS] Checking document relevance with LLM...")
    
    if not documents:
        print("[RAG: GRADE] No documents found. Routing to search fallback.")
        return {"documents": [], "web_search": True}

    llm = get_llm()
    filtered_docs: List[Document] = []
    web_search = False

    grade_prompt = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "Here is the retrieved document:\n\n{context}\n\n"
        "Here is the user question: {question}\n"
        "If the document contains information related to the question, grade as 'yes'. Otherwise 'no'.\n"
        "Respond ONLY with 'yes' or 'no'."
    )

    for doc in documents:
        try:
            prompt_text = grade_prompt.format(context=doc.page_content, question=question)
            response = llm.invoke([HumanMessage(content=prompt_text)])
            grade = response.content.strip().lower()
            
            if "yes" in grade:
                print(f"[GRADE] '{doc.metadata.get('source', 'doc')}' -> RELEVANT (yes)")
                filtered_docs.append(doc)
            else:
                print(f"[GRADE] '{doc.metadata.get('source', 'doc')}' -> NOT RELEVANT (no)")
        except Exception:
            filtered_docs.append(doc)

    if not filtered_docs:
        print("[GRADE] No relevant documents found. Setting web_search = True")
        web_search = True

    return {"documents": filtered_docs, "web_search": web_search}


def transform_query(state: UnifiedAgentState) -> Dict[str, Any]:
    """Rewrites the query to optimize it for external search."""
    question = state["question"]
    iterations = state.get("iterations", 0) + 1
    
    print(f"\n[RAG: TRANSFORM_QUERY] Rewriting query (Attempt #{iterations})...")
    
    llm = get_llm()
    rewrite_prompt = (
        "Rewrite the following user question to make it an effective, concise search engine query.\n"
        f"Question: {question}\n"
        "Provide ONLY the rewritten query text:"
    )
    
    try:
        response = llm.invoke([HumanMessage(content=rewrite_prompt)])
        better_question = response.content.strip().replace('"', '')
        print(f"[RAG: TRANSFORM_QUERY] Rewritten: '{better_question}'")
    except Exception:
        better_question = question

    return {"question": better_question, "iterations": iterations}


def web_search_node(state: UnifiedAgentState) -> Dict[str, Any]:
    """Performs web search using DuckDuckGo when internal documents are insufficient."""
    question = state["question"]
    documents = state.get("documents", [])
    
    print(f"\n[RAG: WEB_SEARCH] Executing web search for: '{question}'")
    
    try:
        search_results = search_web.invoke({"query": question})
        web_doc = Document(
            page_content=search_results,
            metadata={"source": "DuckDuckGo Web Search"}
        )
        documents = list(documents) + [web_doc]
    except Exception as e:
        print(f"[RAG: WEB_SEARCH ERROR] {e}")

    return {"documents": documents}


def generate(state: UnifiedAgentState) -> Dict[str, Any]:
    """Synthesizes the final answer using retrieved context and citations."""
    question = state["question"]
    documents = state.get("documents", [])
    
    print("\n[RAG: GENERATE] Synthesizing final answer with citations...")

    
    context_blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", None)
        loc = f"{source} (Page {page + 1})" if page is not None else source
        context_blocks.append(f"[{i}] Source: {loc}\n{doc.page_content}\n")
    
    context_str = "\n---\n".join(context_blocks) if context_blocks else "No relevant documents found."

    system_instruction = (
        "You are an intelligent knowledge assistant.\n"
        "Answer the user's question accurately using ONLY the context provided below.\n"
        "Cite your sources (e.g., `[sample.pdf]` or `[DuckDuckGo Web Search]`).\n"
        "If you do not know the answer or the context doesn't contain it, clearly state that you don't know.\n\n"
        f"Context:\n{context_str}"
    )

    llm = get_llm()
    prompt = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=question),
    ]

    response = llm.invoke(prompt)
    return {
        "generation": response.content,
        "messages": [AIMessage(content=response.content)],
    }


def decide_to_generate(state: UnifiedAgentState) -> Literal["transform_query", "generate"]:
    """Determines whether to proceed to answer generation or rewrite the query for web search."""
    web_search = state.get("web_search", False)
    iterations = state.get("iterations", 0)
    question = state.get("question", "").lower()

    # If the user is asking about an internal document or specific page, do not search the external web
    is_doc_specific = any(term in question for term in ["page", "pdf", "doc", "document", "file", "guide", "slide"])

    if web_search and iterations < 1 and not is_doc_specific:
        return "transform_query"
    return "generate"


# ---------------------------------------------------------
# Unified Graph Assembly
# ---------------------------------------------------------
def create_unified_agent_graph(checkpointer: MemorySaver = None):
    """Compiles the Unified Adaptive Graph with Autonomous Intent Routing."""
    if checkpointer is None:
        checkpointer = MemorySaver()

    workflow = StateGraph(UnifiedAgentState)

    # 1. Add Router & Decision Node
    workflow.add_node("router", router_node)

    # 2. Add ReAct Branch Nodes
    workflow.add_node("react_agent", react_agent_node)
    tool_node = ToolNode(agent_tools)
    workflow.add_node("tools", tool_node)

    # 3. Add RAG Branch Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate)

    # Entry point is always the Router
    workflow.set_entry_point("router")

    # Router conditionally routes to either RAG or ReAct
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "react_agent": "react_agent",
        },
    )

    # ReAct Branch flow
    workflow.add_conditional_edges(
        "react_agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "react_agent")

    # RAG Branch flow
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=checkpointer)


# Backward compatibility aliases
create_rag_graph = create_unified_agent_graph
create_adaptive_agent = create_unified_agent_graph
