# agent/alerter.py
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

Settings.llm = Ollama(
    model="llama3",
    request_timeout=180.0,
    additional_kwargs={"options": {"num_gpu": 28, "num_ctx": 2048, "num_thread": 4}},
)
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"options": {"num_gpu": 0}},
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")


def _get_or_build_knowledge_index() -> VectorStoreIndex:
    collection = chroma_client.get_or_create_collection("nutriscan_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    knowledge_dir = os.path.join(os.path.dirname(__file__), "..", "Database")

    if collection.count() > 0:
        return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    if not os.path.exists(knowledge_dir):
        raise FileNotFoundError(f"Database/skin_data.txt not found at: {knowledge_dir}")

    docs = SimpleDirectoryReader(knowledge_dir, required_exts=[".txt"]).load_data()
    return VectorStoreIndex.from_documents(docs, storage_context=storage_context, show_progress=True)


knowledge_index = _get_or_build_knowledge_index()

ALERT_PROMPT = PromptTemplate("""
You are NutriScan Guardian, a friendly preventive health companion.
You are NOT a doctor. You give science-backed awareness nudges only.

USER PAST HISTORY (past similar events from memory):
{user_history}

MEDICAL KNOWLEDGE:
{knowledge_context}

CURRENT BODY READINGS (based on ALL {total_readings} sensor readings):
{current_readings}

PERSONAL BASELINE (computed from all {total_readings} readings — this is what is NORMAL for THIS user):
{baseline_summary}

TREND (last 10 readings vs overall average):
{trend_summary}

ANOMALIES DETECTED:
{anomalies}

MEAL RECENTLY CONSUMED:
{meal}

Based on all the above, write a short, friendly alert:
1. What is happening in the body right now (1-2 sentences, simple language).
2. How this specifically affects their skin (acne, oiliness, redness, dullness).
3. Whether this is a PERSONAL anomaly for this user (sigma scores) or just a general threshold breach.
4. Whether the trend is worsening or recovering.
5. Give exactly 2 specific actions they can take RIGHT NOW.
6. End with: Skin Risk: [LOW / MODERATE / HIGH] — one-sentence reason.

Keep it under 160 words. Tone: supportive coach, not scary doctor.
""")


def generate_alert(
    user_id: str,
    current_data: dict,
    anomalies: list,
    meal: str = "Not recorded",
    baseline: dict = None,
    trend: dict = None,
    total_readings: int = 0,
) -> str:
    from agent.memory import retrieve_user_history

    anomaly_query = " ".join([a["metric"].replace("_", " ") for a in anomalies])
    if meal and meal != "Not recorded":
        anomaly_query += f" {meal}"

    knowledge_retriever = knowledge_index.as_retriever(similarity_top_k=3)
    knowledge_nodes = knowledge_retriever.retrieve(anomaly_query)
    knowledge_context = "\n".join([n.text for n in knowledge_nodes])

    user_history = retrieve_user_history(user_id, anomaly_query)

    # ── Format current readings ──
    reading_lines = [
        f"  - HRV Drop:              {current_data.get('hrv_drop_pct', 0):.1f}%",
        f"  - BVP Intensity:         {current_data.get('bvp_intensity_pct', 0):.1f}%",
        f"  - Pulse Amp Change:      {current_data.get('pulse_amp_change_pct', 0):.1f}%",
        f"  - HR Peak:               {current_data.get('hr_peak_pct', 0):.1f}%",
        f"  - Inflammation Watch:    {current_data.get('inflammation_watch_mins', 0)} mins",
        f"  - Spike Duration:        {current_data.get('spike_duration_mins', 0)} mins",
        f"  - Latest Spike Index:    {current_data.get('latest_spike_index', 0):.1f} / 100",
        f"  - Avg Spike Index:       {current_data.get('avg_spike_index', 0):.1f} / 100",
        f"  - Latest SI Percentile:  {current_data.get('latest_si_percentile', 0):.1f}% (vs all readings)",
        f"  - Max Consecutive Spike: {current_data.get('max_consecutive_spike', 0)} readings",
        f"  - HRV Sigma:             {current_data.get('hrv_sigma', 0):.2f}σ from personal normal",
        f"  - BVP Sigma:             {current_data.get('bvp_sigma', 0):.2f}σ from personal normal",
    ]

    # ── Format baseline ──
    if baseline:
        baseline_lines = [
            f"  - Avg HR:   {baseline.get('b_hr', 0):.1f} bpm  (±{baseline.get('sd_hr', 0):.1f})",
            f"  - Avg HRV:  {baseline.get('b_hrv', 0):.1f}     (±{baseline.get('sd_hrv', 0):.1f})",
            f"  - Avg Temp: {baseline.get('b_temp', 0):.2f}°C  (±{baseline.get('sd_temp', 0):.2f})",
            f"  - Avg BVP:  {baseline.get('b_bvp', 0):.3f}    (±{baseline.get('sd_bvp', 0):.3f})",
            f"  - Based on: {baseline.get('total_readings_used', 0)} readings",
        ]
        baseline_summary = "\n".join(baseline_lines)
    else:
        baseline_summary = "Not available"

    # ── Format trend ──
    if trend:
        def trend_label(val):
            if val is None: return "n/a"
            if val > 5:  return f"+{val:.1f}% ↑ (worsening)"
            if val < -5: return f"{val:.1f}% ↓ (recovering)"
            return f"{val:.1f}% → (stable)"

        trend_lines = [
            f"  - HRV trend:   {trend_label(trend.get('hrv_trend_pct'))}",
            f"  - BVP trend:   {trend_label(trend.get('bvp_trend_pct'))}",
            f"  - Spike trend: {trend_label(trend.get('spike_trend_pct'))}",
            f"  - Temp trend:  {trend_label(trend.get('temp_trend_pct'))}",
        ]
        trend_summary = "\n".join(trend_lines)
    else:
        trend_summary = "Not available"

    # ── Format anomalies ──
    anomalies_text = "\n".join([
        f"  - [{a['severity']}] {a['metric']}: {a['value']} → {a['skin_effect']}"
        for a in anomalies
    ]) if anomalies else "None — all values within normal range."

    prompt = ALERT_PROMPT.format(
        user_history=user_history,
        knowledge_context=knowledge_context,
        current_readings="\n".join(reading_lines),
        baseline_summary=baseline_summary,
        trend_summary=trend_summary,
        anomalies=anomalies_text,
        meal=meal,
        total_readings=total_readings,
    )

    response = Settings.llm.complete(prompt)
    return str(response).strip()