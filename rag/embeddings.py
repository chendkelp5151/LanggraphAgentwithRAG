"""Embedding provider supporting local FastEmbed (ONNX, free & fast) and NVIDIA NIM Embeddings."""

import os
from typing import Optional
from langchain_core.embeddings import Embeddings
import config

# Disable symlinks warning on Windows for HuggingFace/FastEmbed cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def get_embeddings() -> Embeddings:
    """Returns embeddings for document indexing and retrieval.
    
    Priority:
    1. FastEmbed (local ONNX BAAI/bge-small-en-v1.5) - 100% free, fast, no external API quota or latency.
    2. NVIDIA NIM Embeddings (if explicitly requested).
    3. FakeEmbeddings (as safety fallback).
    """
    # Priority 1: FastEmbed (Local, High Quality, Free)
    try:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    except Exception as e:
        print(f"[INFO] FastEmbed not available: {e}. Checking alternatives...")

    # Priority 2: NVIDIA NIM Embeddings
    api_key = config.NVIDIA_API_KEY
    if api_key and not api_key.startswith("nvapi-your"):
        try:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
            return NVIDIAEmbeddings(
                model="nvidia/llama-3.2-nv-embedqa-1b-v2",
                api_key=api_key,
                truncate="END"
            )
        except Exception:
            pass

    # Priority 3: Fallback
    from langchain_community.embeddings import FakeEmbeddings
    print("[WARN] Using FakeEmbeddings fallback.")
    return FakeEmbeddings(size=384)
