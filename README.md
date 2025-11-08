# Digital Twin Recruiter Query UI

A lightweight web interface for exercising recruiter-style questions against the Digital Twin RAG system.

## Features
- Ask endpoint with behavioral + PII detection flags
- Sample query catalog (40 queries) with filter buttons (All / Behavioral / PII)
- Semantic search explorer (shows top chunks & relevance scores)
- Displays which chunks were retrieved for each answer

## Backend
Implemented using FastAPI in `digitaltwin_api.py`.

### Endpoints
- `GET /api/health` – readiness check
- `GET /api/sample-queries` – returns catalog
- `GET /api/search?q=...` – semantic search (optional category/tag filtering via query params)
- `POST /api/ask` – answer generation with guardrails

## Run Locally
```bash
pip install -r requirements.txt
uvicorn digitaltwin_api:app --reload
```
Then open `web/index.html` in a browser (or serve statically with any simple file server). If running from the project root with a typical dev server, relative `/api/*` calls will hit the FastAPI instance.

## Environment Variables
Ensure `.env` contains:
```
UPSTASH_VECTOR_REST_URL=...
UPSTASH_VECTOR_REST_TOKEN=...
GROQ_API_KEY=...
```

## Notes
- PII requests (email/phone/address/etc.) return a guarded response.
- Behavioral queries trigger STAR chunk prioritization.
- Retrieved chunk titles help you audit relevance.
- Front-end is framework-free (plain HTML/CSS/JS) for simplicity.

## Next Ideas
- Add chunk preview modal
- Add latency metrics and token usage
- Include answer grading rubric overlay
- Export Q/A transcript
