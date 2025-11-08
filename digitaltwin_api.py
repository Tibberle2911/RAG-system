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

# Serve the UI from /web and redirect root to the UI
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/web/index.html")

@app.get("/about")
async def about_redirect():
    return RedirectResponse(url="/web/about.html")

@app.get("/github")
async def github_redirect():
    return RedirectResponse(url="/web/github.html")

@app.get("/testing")
async def testing_redirect():
    return RedirectResponse(url="/web/testing.html")

@app.get("/profile-data")
async def profile_data_redirect():
    return RedirectResponse(url="/web/profile-data.html")

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
    return {"status": "ok", "index_ready": INDEX is not None, "groq_ready": GROQ is not None}

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

# Run with: uvicorn digitaltwin_api:app --reload
