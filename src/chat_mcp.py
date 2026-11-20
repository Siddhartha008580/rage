"""Compatibility launcher for the unified RAG chat with MCP tool routing."""
try:
    from .chat_rag import main
except ImportError:
    from chat_rag import main


if __name__ == "__main__":
    main()
