# agent/memory.py
import json
from datetime import datetime
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Shared persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def store_user_event(user_id: str, event: dict):
    """
    Store one IoT event + anomalies into the user's personal ChromaDB collection.
    Each user gets their own collection: user_{user_id}_history
    This is what gives the agent its memory of past patterns.

    Args:
        user_id: Firestore user ID (e.g. "user123")
        event:   Dict containing delta readings + anomalies + meal
    """
    collection = chroma_client.get_or_create_collection(f"user_{user_id}_history")

    # Convert event to a natural language memory string the LLM can reason over
    anomaly_summary = json.dumps(event.get("anomalies", []), indent=2)
    text = f"""
Date/Time: {datetime.now().isoformat()}
Timestamp in session: {event.get('timestamp', 'unknown')}

Body Readings:
  - HRV Drop:         {event.get('hrv_drop_pct', 0):.1f}%
  - BVP Intensity:    {event.get('bvp_intensity_pct', 0):.1f}%
  - Pulse Amp Change: {event.get('pulse_amp_change_pct', 0):.1f}%
  - HR Peak:          {event.get('hr_peak_pct', 0):.1f}%
  - Skin Temp Rise:   {event.get('skin_temp_rise', 0):.2f}°C
  - Spike Index:      {event.get('spike_index', 0):.1f}

Meal Logged: {event.get('meal') or 'Not recorded'}
Risk Level: {event.get('risk_level', 'UNKNOWN')}

Anomalies Detected:
{anomaly_summary}
""".strip()

    doc = Document(
        text=text,
        metadata={
            "user_id":   user_id,
            "stored_at": datetime.now().isoformat(),
            "risk_level": event.get("risk_level", "UNKNOWN"),
            "meal":      event.get("meal") or "unknown",
        }
    )

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents([doc], storage_context=storage_context)


def retrieve_user_history(user_id: str, query: str, top_k: int = 5) -> str:
    """
    Retrieve the most semantically relevant past events for a user.
    Used to inject personalised history into the LLM prompt.

    Args:
        user_id: Firestore user ID
        query:   The anomaly/meal string to search against
        top_k:   Number of past events to retrieve

    Returns:
        Formatted string of past events, or a fallback message.
    """
    try:
        collection = chroma_client.get_or_create_collection(f"user_{user_id}_history")

        # No history yet for this user
        if collection.count() == 0:
            return "No past history found. This appears to be the user's first tracked event."

        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Load existing index (no documents — already stored)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context
        )

        retriever = index.as_retriever(similarity_top_k=top_k)
        results = retriever.retrieve(query)

        if not results:
            return "No relevant past events found for this query."

        sections = [f"Past Event {i+1}:\n{r.text}" for i, r in enumerate(results)]
        return "\n\n---\n\n".join(sections)

    except Exception as e:
        return f"Could not retrieve history: {str(e)}"


def get_user_event_count(user_id: str) -> int:
    """Returns how many events are stored for a user."""
    try:
        collection = chroma_client.get_or_create_collection(f"user_{user_id}_history")
        return collection.count()
    except Exception:
        return 0


def clear_user_history(user_id: str):
    """
    Wipe all stored history for a user.
    Use carefully — only expose this in admin/debug routes.
    """
    try:
        chroma_client.delete_collection(f"user_{user_id}_history")
        return True
    except Exception:
        return False