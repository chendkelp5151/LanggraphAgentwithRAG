"""Interactive Command Line Chat Agent with Autonomous Adaptive Routing (LangGraph)."""

import os
import sys
import uuid

# Ensure UTF-8 output encoding for emojis on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langchain_core.messages import HumanMessage, ToolMessage
import config
from agent import create_adaptive_agent
from rag.ingest import ingest_documents, DOCS_DIR

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    USER_COLOR = Fore.CYAN + Style.BRIGHT
    BOT_COLOR = Fore.GREEN + Style.BRIGHT
    TOOL_COLOR = Fore.YELLOW
    ROUTER_COLOR = Fore.BLUE + Style.BRIGHT
    ERR_COLOR = Fore.RED + Style.BRIGHT
    INFO_COLOR = Fore.MAGENTA
    RESET = Style.RESET_ALL
except ImportError:
    USER_COLOR = ""
    BOT_COLOR = ""
    TOOL_COLOR = ""
    ROUTER_COLOR = ""
    ERR_COLOR = ""
    INFO_COLOR = ""
    RESET = ""

def print_banner():
    banner = f"""
===============================================================
    NVIDIA Nemotron 3 Ultra - Unified Autonomous Agent         
===============================================================
 Model       : {config.NVIDIA_MODEL}
 Architecture: LangGraph Adaptive Router (RAG + ReAct Tools)
 Routing     : Autonomous (Model decides RAG vs Tools per query)
 Commands    : /ingest, /help, /clear, /tools, /exit
===============================================================
"""
    print(INFO_COLOR + banner + RESET)

def print_help():
    print(INFO_COLOR + """
Available commands:
  /ingest  - Ingest & index all documents inside data/docs/
  /help    - Show this help message
  /clear   - Reset conversation memory (starts a new thread)
  /tools   - List available agent tools
  /exit    - Exit the chat agent
""" + RESET)

def main():
    print_banner()

    if not config.validate_config():
        sys.exit(1)

    print(INFO_COLOR + "Initializing Autonomous Adaptive Agent and memory..." + RESET)
    try:
        agent = create_adaptive_agent()
    except Exception as e:
        print(ERR_COLOR + f"Failed to initialize agent: {e}" + RESET)
        sys.exit(1)

    thread_id = str(uuid.uuid4())
    agent_config = {"configurable": {"thread_id": thread_id}}
    print(INFO_COLOR + f"Chat session started (Thread: {thread_id[:8]}). Type your message below.\n" + RESET)

    while True:
        try:
            user_input = input(f"{USER_COLOR}You: {RESET}").strip()
            
            if not user_input:
                continue

            # Command handling
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print(INFO_COLOR + "\nGoodbye! Have a great day!" + RESET)
                break
            elif user_input.lower() == "/help":
                print_help()
                continue
            elif user_input.lower() == "/ingest":
                print(INFO_COLOR + f"\nIngesting documents from '{DOCS_DIR}'..." + RESET)
                count = ingest_documents()
                print(INFO_COLOR + f"Indexed {count} chunk(s) into ChromaDB.\n" + RESET)
                continue
            elif user_input.lower() == "/clear":
                thread_id = str(uuid.uuid4())
                agent_config = {"configurable": {"thread_id": thread_id}}
                print(INFO_COLOR + f"\n[Session cleared! New thread: {thread_id[:8]}]\n" + RESET)
                continue
            elif user_input.lower() == "/tools":
                from tools import agent_tools
                print(INFO_COLOR + "\nAvailable tools:" + RESET)
                for tool in agent_tools:
                    print(f" - {TOOL_COLOR}{tool.name}{RESET}: {tool.description.splitlines()[0]}")
                print()
                continue

            inputs = {
                "question": user_input,
                "messages": [HumanMessage(content=user_input)],
                "iterations": 0
            }
            last_generation = ""

            for event in agent.stream(inputs, config=agent_config):
                for node_name, node_output in event.items():
                    if node_name == "router":
                        route = node_output.get("route", "react").upper()
                        print(f"{ROUTER_COLOR}[🧠 Autonomous Router] Determined intent -> '{route}'{RESET}")
                    elif node_name == "retrieve":
                        docs = node_output.get("documents", [])
                        print(f"{TOOL_COLOR}[📥 Retrieve] Found {len(docs)} chunk(s) in ChromaDB{RESET}")
                    elif node_name == "grade_documents":
                        docs = node_output.get("documents", [])
                        needs_search = node_output.get("web_search", False)
                        if needs_search:
                            print(f"{TOOL_COLOR}[⚖️ Grade] Chunks not relevant -> Triggering fallback{RESET}")
                        else:
                            print(f"{TOOL_COLOR}[⚖️ Grade] {len(docs)} chunk(s) confirmed relevant{RESET}")
                    elif node_name == "transform_query":
                        print(f"{TOOL_COLOR}[🔄 Transform Query] Rewrote query -> '{node_output.get('question')}'{RESET}")
                    elif node_name == "web_search":
                        print(f"{TOOL_COLOR}[🌐 Web Search] Fallback search executed{RESET}")
                    elif node_name == "react_agent":
                        msgs = node_output.get("messages", [])
                        if msgs and hasattr(msgs[-1], "tool_calls") and msgs[-1].tool_calls:
                            for tc in msgs[-1].tool_calls:
                                print(f"{TOOL_COLOR}[🔧 Calling Tool: {tc['name']} with args {tc['args']}]{RESET}")
                        elif msgs and hasattr(msgs[-1], "content") and msgs[-1].content:
                            last_generation = msgs[-1].content
                    elif node_name == "tools":
                        msgs = node_output.get("messages", [])
                        for m in msgs:
                            if isinstance(m, ToolMessage):
                                print(f"{TOOL_COLOR}[🔧 Tool '{m.name}' returned]: {m.content[:120]}...{RESET}")
                    elif node_name == "generate":
                        last_generation = node_output.get("generation", "")

            if last_generation:
                print(f"{BOT_COLOR}Nemotron: {RESET}{last_generation}\n")
            else:
                print()

        except KeyboardInterrupt:
            print(INFO_COLOR + "\n\nSession terminated by user." + RESET)
            break
        except Exception as e:
            print(f"\n{ERR_COLOR}An error occurred: {e}{RESET}\n")

if __name__ == "__main__":
    main()
