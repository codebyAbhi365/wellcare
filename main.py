# main.py
from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import shutil
import whisper
import wave
import json
import subprocess
from pathlib import Path

# Initialize Whisper model (Tiny for speed, can be upgraded to base/small)
stt_model = whisper.load_model("base")

from firebase.reader import get_agent_context
from agent.analyzer import analyze_readings, get_overall_risk
from agent.alert import generate_alert, knowledge_index, Settings
from agent.memory import store_user_event, retrieve_user_history
from agent.ml_pipeline import ml_router
from food_recommend_engine.router import recommend_router
from spike_engine import process_and_push


app = FastAPI(title="NutriScan Guardian Agent", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
# Piper TTS Configuration
PIPER_EXE = r"C:\Users\abhishek dipak kadam\AppData\Roaming\Python\Python312\Scripts\piper.exe"
VOICE_MODEL_PATH = BASE_DIR / "voice_agent_aleena" / "en_GB-alba-medium.onnx"

# Setup Static Files for Audio Playback
STATIC_DIR = Path("static")
AUDIO_DIR = STATIC_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(ml_router)
app.include_router(recommend_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AlertRequest(BaseModel):
    user_id: str = "user123"
    meal_logged: Optional[str] = None

class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.post("/process_spike/{user_id}")
def process_spike(user_id: str):
    """Recomputes spike_index for all readings and pushes back to Firebase"""
    result = process_and_push(user_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/agent/alert")
async def get_alert(request: AlertRequest, background_tasks: BackgroundTasks):
    """
    Full pipeline using ALL readings:
    Firebase (all readings) → your exact formulas → sigma scoring
    → trend detection → anomaly detection → RAG alert
    """
    ctx = get_agent_context(request.user_id)
    if "error" in ctx:
        raise HTTPException(status_code=404, detail=ctx["error"])

    metrics  = ctx["metrics"]
    baseline = ctx["baseline"]
    trend    = ctx["trend"]
    total    = ctx["total_readings"]

    anomalies  = analyze_readings(metrics)
    risk_level = get_overall_risk(anomalies)

    if not anomalies:
        return {
            "risk_level": "LOW",
            "alert": "Your body signals look stable right now. Keep it up!",
            "anomalies": [],
            "metrics": metrics,
            "total_readings_analysed": total,
        }

    alert_text = generate_alert(
        user_id=request.user_id,
        current_data=metrics,
        anomalies=anomalies,
        meal=request.meal_logged or "Not recorded",
        baseline=baseline,
        trend=trend,
        total_readings=total,
    )

    event = {**metrics, "anomalies": anomalies,
             "meal": request.meal_logged, "risk_level": risk_level}
    background_tasks.add_task(store_user_event, request.user_id, event)

    return {
        "risk_level":             risk_level,
        "alert":                  alert_text,
        "anomalies":              anomalies,
        "metrics":                metrics,
        "trend":                  trend,
        "total_readings_analysed": total,
    }


@app.post("/agent/chat")
async def chat_with_agent(request: ChatRequest):
    ctx = get_agent_context(request.user_id)
    metrics  = ctx.get("metrics", {})
    trend    = ctx.get("trend", {})
    baseline = ctx.get("baseline", {})
    total    = ctx.get("total_readings", 0)

    current_summary = (
        f"Based on {total} readings — "
        f"HRV Drop: {metrics.get('hrv_drop_pct', 0)}% ({metrics.get('hrv_sigma', 0):.1f}σ from personal normal), "
        f"BVP: {metrics.get('bvp_intensity_pct', 0)}% ({metrics.get('bvp_sigma', 0):.1f}σ), "
        f"Spike Index: {metrics.get('latest_spike_index', 0)} "
        f"(top {100 - metrics.get('latest_si_percentile', 0):.0f}% of all readings), "
        f"Trend: HRV {trend.get('hrv_trend_pct', 0):+.1f}%, Spike {trend.get('spike_trend_pct', 0):+.1f}%."
    ) if "error" not in ctx else "No data available."

    user_history = retrieve_user_history(request.user_id, request.message)
    knowledge_nodes = knowledge_index.as_retriever(similarity_top_k=3).retrieve(request.message)
    knowledge_context = "\n".join([n.text for n in knowledge_nodes])

    prompt = f"""
You are NutriScan Guardian. Answer the user's question using their full history and knowledge.

CURRENT STATE ({total} readings analysed):
{current_summary}

USER HISTORY:
{user_history}

MEDICAL KNOWLEDGE:
{knowledge_context}

USER ASKS: {request.message}

Be friendly, specific, max 120 words. Reference their personal sigma scores if relevant. End with one actionable tip.
"""
    response = Settings.llm.complete(prompt)
    return {"response": str(response)}


@app.post("/agent/chat")
async def chatbot_response(request: dict):
    """Simple chatbot response with welcome message"""
    user_msg = request.get("message", "").lower()
    
    if "hello" in user_msg or "hi" in user_msg:
        response = "Welcome to WellCare! I am Aleena, your personal health assistant. I can help you monitor your metabolic health and give you insights based on your body readings. How can I help you today?"
    else:
        response = f"I heard you say: '{user_msg}'. I'm currently learning more about your health reports to give you a better answer!"
    
    return {"response": response}

@app.post("/agent/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Transcribes uploaded audio using local Whisper model"""
    temp_path = f"temp_{uuid.uuid4()}.m4a"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Transcribe
        result = stt_model.transcribe(temp_path)
        text = result.get("text", "").strip()
        
        print(f"STT Result: {text}")
        return {"text": text}
    except Exception as e:
        print(f"STT Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/agent/tts")
async def text_to_speech(request: dict):
    """Synthesizes speech using local Piper model"""
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    
    try:
        if not VOICE_MODEL_PATH.exists():
            print(f"DEBUG: Model not found at {VOICE_MODEL_PATH}")
            raise HTTPException(status_code=500, detail=f"Model not found at {VOICE_MODEL_PATH}")

        output_filename = f"voice_{uuid.uuid4()}.wav"
        output_path = AUDIO_DIR / output_filename
        
        # Robust subprocess call using the suggested Piper configuration
        command = [
            PIPER_EXE,
            "--model", str(VOICE_MODEL_PATH),
            "--output_file", str(output_path)
        ]

        # The 'shell=True' is crucial on Windows when calling executables in Program Files
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            shell=True 
        )
        
        stdout, stderr = process.communicate(input=text)
        
        if process.returncode != 0:
            print(f"Piper Error (Code {process.returncode}): {stderr}")
            raise HTTPException(status_code=500, detail=f"Piper error: {stderr}")
            
        print(f"DEBUG: TTS Success. Audio saved to {output_path}")
        return {"audio_url": f"/static/audio/{output_filename}"}
        
    except Exception as e:
        print(f"TTS Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "service": "NutriScan Guardian Agent"}

if __name__ == "__main__":
    import uvicorn
    # Use the filename 'main' and the FastAPI instance 'app'
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)