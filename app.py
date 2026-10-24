"""Streamlit Web UI for NVIDIA Nemotron 3 Ultra Autonomous Adaptive Agent."""

import os
import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import config
from agent import create_adaptive_agent
from rag.ingest import index_single_document, get_vectorstore, DOCS_DIR

st.set_page_config(
    page_title="NVIDIA Nemotron 3 Ultra - Autonomous Adaptive Agent",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ NVIDIA Nemotron 3 Ultra - Autonomous Agent")
st.caption(f"Unified LangGraph Agent with Autonomous Intent Routing & RAG ({config.NVIDIA_MODEL})")

# Check API Key
if not config.NVIDIA_API_KEY or config.NVIDIA_API_KEY.startswith("nvapi-your"):
    st.error("⚠️ NVIDIA_API_KEY is not configured! Please set it in your `.env` file.")
    st.stop()

# Initialize session state with version tracking to avoid stale graphs
AGENT_VERSION = "v2_page_aware"

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "agent" not in st.session_state or st.session_state.get("agent_version") != AGENT_VERSION:
    st.session_state.agent = create_adaptive_agent()
    st.session_state.agent_version = AGENT_VERSION

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Agent Settings")
    st.markdown(f"**Model:** `{config.NVIDIA_MODEL}`")
    st.markdown(f"**Session Thread:** `{st.session_state.thread_id[:8]}`")
    st.markdown("**Routing:** 🧠 *Autonomous (LLM Intent Classification)*")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    
    # Knowledge Base / Document Upload Section
    st.subheader("📚 Knowledge Base (Documents)")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files to index:",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Index Uploaded Files", use_container_width=True):
            total_indexed = 0
            os.makedirs(DOCS_DIR, exist_ok=True)
            with st.spinner("Processing & indexing documents..."):
                for uploaded_file in uploaded_files:
                    save_path = os.path.join(DOCS_DIR, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    chunks = index_single_document(save_path)
                    total_indexed += chunks
            st.success(f"Indexed {len(uploaded_files)} file(s) ({total_indexed} chunks) into ChromaDB!")

    # Check vectorstore count
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        st.info(f"📊 Vector Store Status: **{count} chunks** indexed.")
    except Exception:
        st.info("📊 Vector Store: Empty or initializing.")

    st.markdown("---")
    st.markdown("### 🧰 Available Agent Capabilities")
    st.markdown("- 🔍 **RAG Knowledge Base** (PDF & Doc QA)")
    st.markdown("- ⚖️ **Self-Grading & Verification**")
    st.markdown("- 🧮 **Safe Calculator** (Math evaluation)")
    st.markdown("- 🌐 **Live Web Search** (DuckDuckGo)")
    st.markdown("- 🕒 **DateTime & Timezone**")
    st.markdown("- 📝 **Memo & Notes Management**")


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# User prompt input
if prompt := st.chat_input("Ask anything! (The agent will automatically determine if it needs docs, math, search, or notes)"):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process response
    with st.chat_message("assistant"):
        agent_config = {"configurable": {"thread_id": st.session_state.thread_id}}
        status_box = st.status("🧠 Analyzing query intent & routing...", expanded=True)
        response_placeholder = st.empty()
        final_response = ""

        try:
            inputs = {
                "question": prompt,
                "messages": [HumanMessage(content=prompt)],
                "iterations": 0,
            }

            for event in st.session_state.agent.stream(inputs, config=agent_config):
                for node_name, node_output in event.items():
                    if node_name == "router":
                        route = node_output.get("route", "react")
                        if route == "rag":
                            status_box.write("🎯 **Router Decision:** Route $\\rightarrow$ **RAG Knowledge Base** (Query matches document QA)")
                        else:
                            status_box.write("🛠️ **Router Decision:** Route $\\rightarrow$ **General ReAct Tools** (Math, Time, Search, Notes)")

                    elif node_name == "retrieve":
                        docs = node_output.get("documents", [])
                        status_box.write(f"📥 **Retrieved** {len(docs)} document chunks from ChromaDB.")

                    elif node_name == "grade_documents":
                        docs = node_output.get("documents", [])
                        needs_search = node_output.get("web_search", False)
                        if needs_search:
                            status_box.write("⚠️ **Document Grading:** Retrieved docs lacked relevant info. Triggering fallback.")
                        else:
                            status_box.write(f"✅ **Document Grading:** {len(docs)} chunk(s) verified as relevant.")

                    elif node_name == "transform_query":
                        rewritten = node_output.get("question", "")
                        status_box.write(f"🔄 **Query Rewriter:** Refined query to: *'{rewritten}'*")

                    elif node_name == "web_search":
                        status_box.write("🌐 **Web Search:** Pulled fallback context from DuckDuckGo.")

                    elif node_name == "react_agent":
                        msgs = node_output.get("messages", [])
                        if msgs:
                            last = msgs[-1]
                            if hasattr(last, "tool_calls") and last.tool_calls:
                                tool_names = ", ".join([tc['name'] for tc in last.tool_calls])
                                status_box.write(f"🤔 **Invoking Tool(s):** `{tool_names}`")
                            elif hasattr(last, "content") and last.content:
                                final_response = last.content
                                response_placeholder.markdown(final_response)

                    elif node_name == "tools":
                        msgs = node_output.get("messages", [])
                        for m in msgs:
                            if isinstance(m, ToolMessage):
                                status_box.write(f"🔧 **Tool `{m.name}` result:** `{m.content[:80]}...`")

                    elif node_name == "generate":
                        final_response = node_output.get("generation", "")
                        response_placeholder.markdown(final_response)

            status_box.update(label="✅ Response Complete", state="complete", expanded=False)

            if final_response:
                st.session_state.messages.append({"role": "assistant", "content": final_response})

        except Exception as e:
            status_box.update(label="❌ Error", state="error", expanded=True)
            st.error(f"Error during agent execution: {str(e)}")
