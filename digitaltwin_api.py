import os
import json
from typing import List, Dict, Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel

import digitaltwin_rag as rag

# Initialize on startup (lazy to allow simple reloads)
INDEX = None
GROQ = None

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

class SearchResponse(BaseModel):
    results: List[Dict]

class SampleQueriesResponse(BaseModel):
    queries: List[Dict]

class BulkAskItem(BaseModel):
    id: Optional[str] = None
    question: str

class BulkAskRequest(BaseModel):
    items: List[BulkAskItem]

class BulkAskItemResult(BaseModel):
    id: Optional[str] = None
    answer: str

class BulkAskResponse(BaseModel):
    results: List[BulkAskItemResult]

app = FastAPI(title="Digital Twin Recruiter Test API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/index.html")

@app.get("/about")
async def about_redirect():
    return RedirectResponse(url="/about.html")

@app.get("/github")
async def github_redirect():
    return RedirectResponse(url="/github.html")

@app.get("/testing")
async def testing_redirect():
    return RedirectResponse(url="/testing.html")

@app.get("/profile-data")
async def profile_data_redirect():
    return RedirectResponse(url="/profile-data.html")

@app.on_event("startup")
async def startup_event():
    global INDEX, GROQ
    INDEX = rag.setup_vector_database(force_rebuild=False)
    GROQ = rag.setup_groq_client()

@app.get("/api/sample-queries", response_model=SampleQueriesResponse)
async def sample_queries():
    """Return central sample query catalog.

    Tries multiple import paths to remain robust whether executed with or without
    tests package on sys.path.
    """
    QUERY_ENTRIES = []
    try:
        from tests.query_entries import QUERY_ENTRIES as qe  # type: ignore
        QUERY_ENTRIES = qe
    except Exception:
        try:
            from query_entries import QUERY_ENTRIES as qe2  # type: ignore
            QUERY_ENTRIES = qe2
        except Exception:
            pass
    return {"queries": [
        {"id": qid, "text": text, "behavioral": flagged} for qid, text, flagged in QUERY_ENTRIES
    ]}

@app.get("/api/search", response_model=SearchResponse)
async def search(q: str = Query(...), category: Optional[str] = None, tag: Optional[str] = None):
    if INDEX is None:
        return {"results": []}
    results = rag.semantic_search(INDEX, q, top_k=8, category=category, tag=tag)
    return {"results": results}

@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        return {"answer": "Empty question."}
    if GROQ is None and INDEX is None:
        return {"answer": rag.fallback_answer(None, question, top_k=8)}
    if GROQ is None:
        return {"answer": rag.fallback_answer(INDEX, question, top_k=8)}
    if INDEX is None:
        return {"answer": rag.fallback_answer(None, question, top_k=8)}
    pii = rag.is_pii_request(question)
    if pii:
        return {"answer": "I can discuss my professional experience and skills, but I don’t share personal contact or identification details."}
    answer = rag.answer_question(GROQ, INDEX, question, top_k=10)
    return {"answer": answer}

@app.post("/api/bulk-ask", response_model=BulkAskResponse)
async def bulk_ask(req: BulkAskRequest):
    results: List[Dict] = []
    if GROQ is None or INDEX is None:
        return {"results": []}
    for item in req.items:
        q = (item.question or "").strip()
        if not q:
            results.append({
                "id": item.id,
                "answer": "Empty question."
            })
            continue
        if rag.is_pii_request(q):
            results.append({
                "id": item.id,
                "answer": "I can discuss my professional experience and skills, but I don’t share personal contact or identification details."
            })
            continue
        answer = rag.answer_question(GROQ, INDEX, q, top_k=10)
        results.append({
            "id": item.id,
            "answer": answer
        })
    return {"results": results}

@app.get("/api/health")
async def health():
    missing_env = []
    for key in ["UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN", "GROQ_API_KEY"]:
        if not os.getenv(key):
            missing_env.append(key)
    return {
        "status": "ok",
        "index_ready": INDEX is not None,
        "groq_ready": GROQ is not None,
        "missing_env": missing_env,
        "cwd_files": [f for f in os.listdir(os.getcwd()) if f.endswith('.py')][:10]
    }

@app.get("/api/env-status")
async def env_status():
    """Report non-secret deployment readiness of required env vars and connection states.

    Returns booleans for each required variable and whether the index/groq client objects exist.
    Does not expose actual values.
    """
    keys = ["UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN", "GROQ_API_KEY"]
    presence = {k: bool(os.getenv(k)) for k in keys}
    return {
        "env": presence,
        "index_ready": INDEX is not None,
        "groq_ready": GROQ is not None,
        "deployment_mode": os.getenv("VERCEL", "local"),
    }

@app.get("/api/diag")
async def diag():
    """Minimal end-to-end diagnostics for Upstash Vector and Groq.

    - If env vars missing, returns which are missing and does not attempt calls.
    - Upstash: performs a tiny query for a benign token ('ping') with top_k=1
      and reports whether the call returned without exception.
    - Groq: sends a 1-2 token prompt and checks that a response object is returned.
    - Never includes secrets or full model output.
    """
    required = ["UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN", "GROQ_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    report: Dict[str, object] = {"missing_env": missing}
    # Early return if any missing
    if missing:
        report.update({"upstash_ok": False, "groq_ok": False, "note": "Set env vars in Vercel settings."})
        return report
    # Upstash check
    upstash_ok = False
    try:
        if INDEX is None:
            idx = rag.setup_vector_database()
        else:
            idx = INDEX
        if idx is not None:
            _ = rag.query_vectors(idx, "ping", top_k=1)
            upstash_ok = True
    except Exception:
        upstash_ok = False
    report["upstash_ok"] = upstash_ok
    # Groq check
    groq_ok = False
    try:
        client = GROQ or rag.setup_groq_client()
        if client:
            # Minimal call; keep tokens small and avoid returning content
            _ = client.chat.completions.create(
                model=rag.DEFAULT_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                temperature=0.1,
                max_tokens=5,
            )
            groq_ok = True
    except Exception:
        groq_ok = False
    report["groq_ok"] = groq_ok
    # High-level readiness
    report["ready"] = bool(upstash_ok and groq_ok)
    return report

@app.get("/api/profile-data")
async def api_profile_data():
    """Return structured professional profile data with optional behavioral stories.

    - Loads digitaltwin.json
    - Optionally loads star_profile.json
    - Masks email/phone in contact details
    - Provides sections for: personal, experience (flattened), skills, education, projects, goals, methodology_stories.
    """
    base_path = os.path.join(os.path.dirname(__file__), 'data')
    profile_path = os.path.join(base_path, 'digitaltwin.json')
    star_path = os.path.join(base_path, 'star_profile.json')
    if not os.path.exists(profile_path):
        return JSONResponse(status_code=404, content={"error": "profile json missing"})
    try:
        with open(profile_path, 'r', encoding='utf-8') as f:
            profile = json.load(f)
    except Exception:
        return JSONResponse(status_code=500, content={"error": "unable to parse profile json"})
    # Mask contact
    personal = profile.get('personal', {})
    contact = personal.get('contact', {})
    if 'email' in contact:
        contact['email'] = '[redacted email]'
    if 'phone' in contact:
        contact['phone'] = '[redacted phone]'
    # Flatten experience stories without STAR titles
    exp_out = []
    for exp in profile.get('experience', []):
        for story in exp.get('achievements_star', []):
            exp_out.append({
                'company': exp.get('company'),
                'role': exp.get('title'),
                'duration': exp.get('duration'),
                'context': exp.get('company_context'),
                'situation': story.get('situation'),
                'task': story.get('task'),
                'action': story.get('action'),
                'result': story.get('result')
            })
    methodology_stories = []
    if os.path.exists(star_path):
        try:
            with open(star_path, 'r', encoding='utf-8') as sf:
                star_data = json.load(sf)
            methodology_stories = star_data.get('stories', [])
        except Exception:
            methodology_stories = []
    resp = {
        'personal': {k: v for k, v in personal.items() if k != 'contact'},
        'contact': contact,
        'experience_stories': exp_out,
        'skills': profile.get('skills', {}),
        'education': profile.get('education', {}),
        'projects': profile.get('projects_portfolio', []),
        'career_goals': profile.get('career_goals', {}),
        'methodology_stories': methodology_stories,
    }
    return resp

# Mount StaticFiles at root after all dynamic routes so it doesn't mask them
app.mount("/", StaticFiles(directory="web", html=True), name="root_static")

# Run with: uvicorn digitaltwin_api:app --reload
