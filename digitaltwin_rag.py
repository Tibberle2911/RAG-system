"""
Digital Twin RAG Application
Based on Binal's production implementation
- Upstash Vector: Built-in embeddings and vector storage
- Groq: Ultra-fast LLM inference
"""

import os
import json
import argparse
from typing import List, Dict, Optional
from dotenv import load_dotenv
from upstash_vector import Index
from groq import Groq
import re

def robust_load_env():
    """Load .env with encoding fallback (handles UTF-16 saved files)."""
    base_dir = os.path.dirname(__file__)
    env_path = os.path.join(base_dir, '.env')
    if not os.path.exists(env_path):
        load_dotenv()  # fall back to default search
        return
    try:
        load_dotenv(env_path, encoding='utf-8')
        return
    except UnicodeDecodeError:
        # Attempt UTF-16 LE/BE decode then rewrite as UTF-8
        for enc in ('utf-16', 'utf-16-le', 'utf-16-be'):
            try:
                with open(env_path, 'r', encoding=enc) as f:
                    content = f.read()
                # Rewrite normalized UTF-8
                with open(env_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"🔧 Converted .env from {enc} to UTF-8")
                load_dotenv(env_path, encoding='utf-8')
                return
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"⚠️ Failed reading .env with {enc}: {e}")
        print("❌ Unable to decode .env file; please resave as UTF-8 (VS Code: Change File Encoding).")

robust_load_env()

JSON_FILE = "digitaltwin.json"
# Optional expected chunk count (used to decide if re-embedding may be needed)
EXPECTED_MIN_CHUNKS = 10
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
DEFAULT_MODEL = "llama-3.1-8b-instant"
BASE_DIR = os.path.dirname(__file__)
POSSIBLE_JSON_PATHS = [
    os.path.join(BASE_DIR, "data", "digitaltwin.json"),
    os.path.join(BASE_DIR, JSON_FILE),
    JSON_FILE,
]
for _p in POSSIBLE_JSON_PATHS:
    if os.path.exists(_p):
        RESOLVED_JSON_FILE = _p
        break
else:
    RESOLVED_JSON_FILE = JSON_FILE

def setup_groq_client():
    """Setup Groq client"""
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found in .env file")
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized successfully!")
        return client
    except Exception as e:
        print(f"❌ Error initializing Groq client: {e}")
        return None

def validate_upstash_env() -> bool:
    """Validate required Upstash Vector environment variables are present."""
    required = ["UPSTASH_VECTOR_REST_URL", "UPSTASH_VECTOR_REST_TOKEN"]
    missing = [r for r in required if not os.getenv(r)]
    if missing:
        print(f"❌ Missing required Upstash env vars: {', '.join(missing)}")
        print("➡️  Add them to your .env file (see .env.example)")
        return False
    return True

def build_content_chunks(profile: Dict) -> List[Dict]:
    """Derive content chunks from rich profile JSON."""
    chunks: List[Dict] = []
    cid = 0
    def add(title: str, content: str, ctype: str, category: str = "", tags: List[str] = None):
        nonlocal cid
        cid += 1
        chunks.append({
            "id": f"chunk_{cid}",
            "title": title,
            "content": content.strip(),
            "type": ctype,
            "metadata": {"category": category, "tags": tags or []}
        })
    personal = profile.get("personal", {})
    if personal:
        # Include name, title, and location explicitly in the personal summary content
        summary_core = personal.get("summary", "")
        name_val = personal.get("name", "")
        title_val = personal.get("title", "")
        location_val = personal.get("location", "")
        combined_summary = " | ".join([v for v in [name_val, title_val, location_val, summary_core] if v])
        add("Personal Summary", combined_summary, "section", "personal", ["summary", title_val, name_val])
        if personal.get("elevator_pitch"):
            add("Elevator Pitch", personal["elevator_pitch"], "snippet", "personal", ["pitch"])
        contact = personal.get("contact", {})
        if contact:
            contact_lines = []
            for k,v in contact.items():
                if v:
                    contact_lines.append(f"{k.title()}: {v}")
            if contact_lines:
                add("Contact Information", "\n".join(contact_lines), "personal", "personal", ["contact"])
    # Salary & location preferences
    salary_loc = profile.get("salary_location", {})
    if salary_loc:
        sl_parts = []
        mappings = [
            ("current_salary","Current Salary"),
            ("salary_expectations","Salary Expectations"),
            ("location_preferences","Location Preferences"),
            ("relocation_willing","Relocation Willing"),
            ("remote_experience","Remote Experience"),
            ("travel_availability","Travel Availability"),
            ("work_authorization","Work Authorization"),
        ]
        for key,label in mappings:
            val = salary_loc.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                sl_parts.append(f"{label}: {', '.join(map(str,val))}")
            else:
                sl_parts.append(f"{label}: {val}")
        if sl_parts:
            add("Salary & Location", "\n".join(sl_parts), "profile", "salary_location", ["work_authorization","remote_experience"])  
    for exp in profile.get("experience", []):
        tags = exp.get("technical_skills_used", [])
        exp_parts = [
            f"Company: {exp.get('company')}",
            f"Title: {exp.get('title')}",
            f"Duration: {exp.get('duration')}",
            f"Context: {exp.get('company_context','')}",
        ]
        if exp.get('team_structure'):
            exp_parts.append(f"Team Structure: {exp.get('team_structure')}")
        for a in exp.get("achievements_star", []):
            exp_parts.append("STAR -> " + "; ".join([
                f"Situation: {a.get('situation')}",
                f"Task: {a.get('task')}",
                f"Action: {a.get('action')}",
                f"Result: {a.get('result')}"
            ]))
            # Add a dedicated STAR chunk for higher retrieval precision
            star_title = f"STAR - {exp.get('company')} - {exp.get('title')}"
            star_lines = [
                f"Situation: {a.get('situation')}",
                f"Task: {a.get('task')}",
                f"Action: {a.get('action')}",
                f"Result: {a.get('result')}"
            ]
            add(star_title, "\n".join(star_lines), "experience", "experience", ["star","behavioral", exp.get('company',''), exp.get('title','')])
        # Add leadership examples and team size if present
        leadership = exp.get("leadership_examples", [])
        if leadership:
            exp_parts.append("Leadership Examples: " + "; ".join(leadership))
        team_size = exp.get("team_size_managed")
        if team_size:
            exp_parts.append(f"Team Size Managed: {team_size}")
        add(f"Experience - {exp.get('company')}", "\n".join(exp_parts), "experience", "experience", tags)
    skills = profile.get("skills", {})
    tech = skills.get("technical", {})
    lang_lines = []
    focus_lines = []
    for lang in tech.get("programming_languages", []):
        base = f"{lang.get('language')}: {lang.get('proficiency')} (Frameworks: {', '.join(lang.get('frameworks', []))})"
        focus = lang.get('focus')
        if focus:
            base += f" (Focus: {focus})"
            focus_lines.append(f"{lang.get('language')}: {focus}")
        lang_lines.append(base)
    if lang_lines:
        add("Technical Skills", "Programming Languages:\n" + "\n".join(lang_lines), "skills", "skills", ["languages","frameworks"])
    if focus_lines:
        add("Language Focus", "\n".join(focus_lines), "skills", "skills", ["language_focus"])
    # Additional technical skill categories
    extra_skill_categories = [
        ("frontend","Frontend"),
        ("databases","Databases"),
        ("data_visualisation","Data Visualisation"),
        ("testing","Testing"),
        ("ai_tools","AI Tools"),
        ("devops_tooling","DevOps Tooling"),
        ("analytics_research","Analytics & Research"),
    ]
    for key,label in extra_skill_categories:
        items = tech.get(key)
        if items:
            add(f"{label} Skills", ", ".join(items), "skills", "skills", [key])
    # Certifications list (from skills.certifications)
    certs = skills.get("certifications")
    if certs:
        add("Certifications", "; ".join(certs), "skills", "skills", ["certifications"])
    soft = skills.get("soft_skills", [])
    if soft:
        add("Soft Skills", ", ".join(soft), "skills", "skills", ["soft-skills"])
    education = profile.get("education", {})
    if education:
        edu_text = " | ".join(filter(None,[education.get("university"),education.get("degree"),education.get("duration"),f"GPA: {education.get('gpa')}"]))
        add("Education", edu_text, "education", "education", ["degree","gpa"])
        awards = education.get("awards")
        if awards:
            add("Education Awards", "; ".join(awards), "education", "education", ["awards"])
        coursework = education.get("relevant_coursework")
        if coursework:
            add("Relevant Coursework", ", ".join(coursework), "education", "education", ["coursework"])
        thesis = education.get("thesis_project")
        if thesis:
            add("Thesis Project", str(thesis), "education", "education", ["thesis"])
    for proj in profile.get("projects_portfolio", []):
        proj_text = " | ".join(filter(None,[proj.get("name"),proj.get("description"),f"Tech: {', '.join(proj.get('technologies', []))}",f"Impact: {proj.get('impact')}",f"Role: {proj.get('role')}"]))
        add(f"Project - {proj.get('name')}", proj_text, "project", "projects", proj.get("technologies", []))
    goals = profile.get("career_goals", {})
    if goals:
        add("Career Goals", "Short Term: " + goals.get("short_term","") + "\nLong Term: " + goals.get("long_term",""), "goals", "career", ["goals"])
        lf = goals.get("learning_focus")
        if lf:
            add("Learning Focus", ", ".join(lf), "goals", "career", ["learning_focus"])
        inds = goals.get("industries_interested")
        if inds:
            add("Industries Interested", ", ".join(inds), "goals", "career", ["industries"])
    prep = profile.get("interview_prep", {})
    common = prep.get("common_questions", {})
    for category, qs in common.items():
        if isinstance(qs,list):
            add(f"Interview - {category.title()}", "\n".join(qs), "interview", "interview", [category])
        elif isinstance(qs,dict):
            text = "Research Areas:\n" + "\n".join(qs.get("research_areas",[])) + "\nQuestions:\n" + "\n".join(qs.get("preparation_questions",[]))
            add("Interview - Company Research", text, "interview", "interview", ["company-research"])
    # Weakness mitigation
    weaknesses = prep.get("weakness_mitigation", [])
    if weaknesses:
        wm_lines = []
        for w in weaknesses:
            wm_lines.append(f"Weakness: {w.get('weakness')}\nMitigation: {w.get('mitigation')}")
        add("Weakness Mitigation", "\n\n".join(wm_lines), "interview", "interview", ["weakness_mitigation"])
    prof = profile.get("professional_development", {})
    # Split professional development into dedicated chunks
    recent_learning = prof.get("recent_learning")
    if recent_learning:
        add("Recent Learning", ", ".join(recent_learning), "development", "development", ["learning"])
    open_source = prof.get("open_source")
    if open_source:
        add("Open Source", "; ".join(open_source), "development", "development", ["open_source"])
    courses = prof.get("courses_certifications")
    if courses:
        add("Courses & Certifications", "; ".join(courses), "development", "development", ["courses","certifications"])
    conferences = prof.get("conferences_attended")
    if conferences:
        add("Conferences Attended", "; ".join(conferences), "development", "development", ["conferences"])
    publications = prof.get("publications")
    if publications:
        add("Publications", "; ".join(publications), "development", "development", ["publications"])

    # Behavioral methodology overview chunk (neutral title, still tagged for retrieval)
    add(
        "Behavioral Methodology Overview",
        (
            "I present experience using a structured situation-task-action-result methodology without labeling answers explicitly. "
            "I first outline the context and objective, then clarify my specific responsibilities, detail decisive actions (technical and collaborative), and conclude with quantified impact (performance gains, reliability improvements, efficiency, cost savings, stakeholder outcomes). "
            "This approach keeps responses concise, outcome-focused, and easy for interviewers to follow while highlighting ownership and measurable results."
        ),
        "methodology",
        "experience",
        ["behavioral","methodology","star"]
    )
    return chunks

def load_profile_json(path: str) -> Optional[Dict]:
    """Load profile JSON from disk."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Profile file not found: {path}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error in profile file: {e}")
    return None

def get_canonical_name(default: str = "") -> str:
    """Fetch the canonical personal name from the profile JSON.

    Returns the `personal.name` field if available, otherwise the provided default.
    """
    profile = load_profile_json(RESOLVED_JSON_FILE)
    try:
        name = (profile or {}).get("personal", {}).get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass
    return default

def persist_profile_json(path: str, data: Dict) -> None:
    """Persist augmented profile JSON (best-effort)."""
    try:
        with open(path, 'w', encoding='utf-8') as wf:
            json.dump(data, wf, ensure_ascii=False, indent=2)
        print("💾 Updated profile JSON saved")
    except Exception as e:
        print(f"⚠️ Failed to persist profile JSON: {e}")

def prepare_vectors_from_chunks(chunks: List[Dict]) -> List[tuple]:
    """Convert chunk list into vector tuples expected by Upstash."""
    vectors: List[tuple] = []
    for chunk in chunks:
        enriched_text = f"{chunk['title']}: {chunk['content']}"
        vectors.append((
            chunk['id'],
            enriched_text,
            {
                "title": chunk.get('title',''),
                "type": chunk.get('type',''),
                "content": chunk.get('content',''),
                "category": chunk.get('metadata',{}).get('category',''),
                "tags": chunk.get('metadata',{}).get('tags',[])
            }
        ))
    return vectors

def embed_profile(index: Index, profile: Dict, force: bool = False) -> int:
    """[Disabled] Importing to the vector DB is handled by data/embed_digitaltwin.py.

    This stub remains for backward compatibility but performs no upserts.
    """
    print("⏭️ Embedding is disabled here. Use data/embed_digitaltwin.py to import vectors.")
    return 0

def setup_vector_database(force_rebuild: bool = False) -> Optional[Index]:
    """Setup vector DB connection only (no embedding).

    Returns None if required env vars missing or connection fails.
    """
    try:
        if not validate_upstash_env():
            return None
        index = Index.from_env()
        # Lightweight info fetch for diagnostics; swallow errors.
        try:
            info = index.info()
            current_count = getattr(info, 'vector_count', 0)
            print(f"Upstash vector count: {current_count}")
        except Exception:
            pass
        return index
    except Exception:
        return None

def query_vectors(index, query_text, top_k=3):
    """Query Upstash Vector for similar vectors"""
    try:
        return index.query(data=query_text, top_k=top_k, include_metadata=True)
    except Exception as e:
        print(f"❌ Error querying vectors: {e}")
        return None

def semantic_search(index: Index, query: str, top_k: int = 8, category: Optional[str] = None, tag: Optional[str] = None):
    """Perform semantic search with optional category / tag filtering.

    Returns list of dict items with id, title, score, content, category, tags.
    """
    raw = query_vectors(index, query, top_k=top_k)
    results = []
    if not raw:
        return results
    for r in raw:
        md = getattr(r, 'metadata', {}) or {}
        # Exclude explicit contact info from retrieval for safety
        if (md.get('title') or '').strip().lower() == 'contact information':
            continue
        if category and md.get('category') != category:
            continue
        tags = md.get('tags') or []
        if tag and tag not in tags:
            continue
        results.append({
            'id': getattr(r, 'id', None),
            'title': md.get('title'),
            'score': getattr(r, 'score', 0.0),
            'content': md.get('content'),
            'category': md.get('category'),
            'tags': tags
        })
    return results

PII_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PII_PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4})")

def sanitize_text(text: str) -> str:
    """Mask likely PII patterns in text (emails, phone-like numbers)."""
    if not text:
        return text
    masked = PII_EMAIL_RE.sub("[redacted email]", text)
    masked = PII_PHONE_RE.sub(lambda m: "[redacted phone]" if len(re.sub(r"\D","",m.group(0)))>=7 else m.group(0), masked)
    return masked

def limit_words(text: str, min_words: int = 50, max_words: int = 200) -> str:
    """Clamp text to a target word range by trimming to max words.

    We enforce the upper bound strictly and guide the lower bound via prompts.
    """
    if not text:
        return text
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
        text = " ".join(words)
    # Ensure text ends at a sentence boundary; trim trailing partial sentence if it was cut.
    sentence_end = re.search(r'[.!?](?:[\'”"])?\s*$', text)
    if not sentence_end:
        # Try to find the last sentence terminator.
        m = re.findall(r'[.!?](?:[\'”"])?', text)
        if m:
            last_idx = max(text.rfind(token) for token in m)
            text = text[: last_idx + 1]
    return text

def _strip_known_prefixes(line: str) -> str:
    """Remove decorative heading prefixes and generic short headings ending with ':' or ' -'."""
    prefixes = [
        r"^\s*Elevator Pitch\s*[:\-]\s*",
        r"^\s*Behavioral Questions\s*[:\-]\s*",
        r"^\s*Company Research\s*[:\-]\s*",
    ]
    for pat in prefixes:
        line = re.sub(pat, "", line, flags=re.IGNORECASE)
    # Generic: short heading before colon, keep the remainder
    line = re.sub(r"^\s*[A-Za-z][^:]{0,40}:\s*", "", line)
    return line

def bulletize_if_needed(text: str) -> str:
    """If no bullet list present, convert sentences after the first into bullets.

    Keeps the first sentence as an intro and turns subsequent sentences into '- ' bullets.
    """
    if re.search(r"(?m)^\s*\-\s+", text):
        return text
    # Split into sentences (simple heuristic)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z'\"(])", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1:
        return text
    intro = sentences[0]
    bullets = sentences[1:]
    # Merge very short sentences with the previous bullet
    merged = []
    for s in bullets:
        if merged and len(s) < 20:
            merged[-1] = merged[-1].rstrip(".") + "; " + s
        else:
            merged.append(s)
    lines = [intro, ""] + [f"- {b}" for b in merged]
    return "\n".join(lines)

def clean_prologue(text: str) -> str:
    """Remove generic meta disclaimers and question restatements at the beginning.

    Examples removed:
    - "Here are the answers..."
    - "In the format requested..."
    - Leading lines starting with: Describe/Tell me about/What is/How do you/Give me/Walk me through/Explain/Summarize/List/Discuss
    """
    if not text:
        return text
    # Work line-wise first to drop obvious meta lines
    lines = [ln.strip() for ln in text.splitlines()]
    drop_prefixes = (
        r"^here\s+(are|is)\b",
        r"^in\s+the\s+format\s+requested\b",
        r"^the\s+following\b",
        r"^below\b",
        r"^i\s+'?d\s+be\s+happy\s+to\b",
        r"^i\s+would\s+be\s+happy\s+to\b",
        r"^i\s+am\s+happy\s+to\b",
        r"^i\s+'?m\s+happy\s+to\b",
        r"^sure[, ]",
        r"^certainly[, ]",
        r"^of\s+course[, ]",
    )
    out_lines = []
    skipping = True
    for ln in lines:
        if skipping and any(re.search(pat, ln, flags=re.IGNORECASE) for pat in drop_prefixes):
            continue
        skipping = False
        out_lines.append(ln)
    cleaned = "\n".join(out_lines).strip()
    if not cleaned:
        return text
    # Remove leading question-like sentences
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    starters = (
        r"^describe\b", r"^tell\s+me\s+about\b", r"^what\s+is\b", r"^how\s+do\s+you\b",
        r"^give\s+me\b", r"^walk\s+me\s+through\b", r"^explain\b", r"^summarize\b", r"^list\b", r"^discuss\b"
    )
    while sentences and re.search(r"|".join(starters), sentences[0].strip(), flags=re.IGNORECASE):
        sentences.pop(0)
    return " ".join(s.strip() for s in sentences if s.strip())

def remove_standalone_headings(text: str) -> str:
    """Remove lines that look like decorative section headings without content.

    Targets lines like 'My Background and Skills', 'Key Highlights', 'Summary', 'Overview'.
    """
    if not text:
        return text
    heading_eq = re.compile(r"^(my\s+)?(background(\s+and\s+skills)?|skills|summary|overview|key\s+highlights|experience\s+highlights|professional\s+summary|profile)\s*$", re.IGNORECASE)
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            out.append(ln)
            continue
        if s.startswith('- '):
            out.append(ln)
            continue
        if heading_eq.match(s):
            continue
        out.append(ln)
    return "\n".join(out)

def simplify_bullet_headings(text: str) -> str:
    """On bullet lines, remove generic heading phrases before a colon, keeping the content.

    Example: "- Collaborating with engineers to formalize a design system: I worked..." -> "- I worked..."
    Preserves known useful labels like Situation/Task/Action/Result/Impact/Tech/Role/Duration.
    """
    def _simplify_line(ln: str) -> str:
        # Remove leading descriptive phrases before colon unless they are allowed semantic labels.
        simplified = re.sub(
            r'^(\-\s*)(?!(Situation|Task|Action|Result|Impact|Tech|Role|Duration)\b)[^:\n]{3,120}:\s*',
            r'\1', ln, flags=re.IGNORECASE)
        # Guard against double hyphen after removal
        simplified = re.sub(r'^(\-\s*)\-\s*', r'\1', simplified)
        return simplified
    return "\n".join(_simplify_line(ln) if ln.lstrip().startswith('- ') else ln for ln in text.splitlines())

def enforce_first_person_singular(text: str) -> str:
    """Replace plural-first-person forms with singular where appropriate.

    This is heuristic and uses word boundaries to avoid partial matches.
    """
    replacements = [
        (r"\bwe're\b", "I'm"), (r"\bwe\s+are\b", "I am"), (r"\bwe\s+were\b", "I was"),
        (r"\bwe've\b", "I've"), (r"\bwe\s+have\b", "I have"), (r"\bwe\s+had\b", "I had"),
        (r"\bwe\s+can\b", "I can"), (r"\bwe\s+could\b", "I could"), (r"\bwe\s+do\b", "I do"), (r"\bwe\s+did\b", "I did"),
        (r"\bwe\b", "I"), (r"\bourselves\b", "myself"), (r"\bours\b", "mine"), (r"\bour\b", "my"), (r"\bus\b", "me"),
    ]
    out = text
    for pat, rep in replacements:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out

def unify_bullets(text: str) -> str:
    """Merge or remove redundant bullets and order for impact.

    Heuristics:
      - Collapse multiple 'Languages' / tech stack bullets into one.
      - Remove bullets that are strict substrings of another bullet.
      - Prioritize bullets with numbers/percentages first.
    """
    lines = text.splitlines()
    bullet_lines = [ln for ln in lines if ln.lstrip().startswith('- ')]
    if not bullet_lines:
        return text
    # Normalize spacing
    norm = []
    for b in bullet_lines:
        s = re.sub(r"\s+", " ", b.strip())
        norm.append(s)
    # Merge language bullets
    lang_pat = re.compile(r'-\s*(languages?:)?\s*(.+)', re.IGNORECASE)
    lang_items = []
    retained = []
    for b in norm:
        m = lang_pat.match(b)
        if m and any(t in b.lower() for t in ['php','python','c#','java','javascript','node']):
            # Extract comma / slash / pipe separated tokens
            tail = m.group(2)
            parts = re.split(r'[,/|]', tail)
            for p in parts:
                token = p.strip()
                if token:
                    lang_items.append(token)
        else:
            retained.append(b)
    if lang_items:
        dedup_langs = []
        seen = set()
        for l in lang_items:
            key = l.lower()
            if key not in seen:
                seen.add(key)
                dedup_langs.append(l)
        retained.insert(0, '- Languages: ' + ', '.join(dedup_langs))
    # Remove bullets that are substrings of another (after normalization)
    filtered = []
    for b in retained:
        content = b[2:].lower()
        if any(content != o[2:].lower() and content in o[2:].lower() for o in retained):
            continue
        filtered.append(b)
    # Prioritize bullets with numbers/percent signs
    def score(b: str) -> tuple:
        has_num = 1 if re.search(r'\d', b) else 0
        length_penalty = -len(b)
        return (has_num, length_penalty)
    filtered.sort(key=score, reverse=True)
    # Rebuild text
    out_lines = [ln for ln in lines if not ln.lstrip().startswith('- ')]
    # Ensure a blank line before bullets if intro exists
    if out_lines and out_lines[-1].strip() != '':
        out_lines.append('')
    out_lines.extend(filtered)
    return '\n'.join(out_lines)

def is_pii_request(question: str) -> bool:
    """Heuristic check for requests seeking personal contact or sensitive identifiers."""
    if not question:
        return False
    q = question.lower()
    pii_terms = [
        'email', 'phone', 'telephone', 'mobile', 'address', 'home address', 'dob', 'date of birth',
        'id number', 'passport', 'driver', "driver's license", 'license number', 'medicare', 'tax file', 'tfn',
        'bank', 'account number', 'credit card', 'ssn', 'social security'
    ]
    return any(t in q for t in pii_terms)

def is_behavioral_query(question: str) -> bool:
    """Detect STAR/behavioral interview style questions."""
    if not question:
        return False
    q = question.lower()
    triggers = [
        # Canonical behavioral framings
        'tell me about a time', 'situation where', 'how did you handle', 'what did you do',
        # Generic markers that commonly imply STAR stories
        'example', 'challenge', 'overcom', 'handle', 'solv', 'result', 'impact', 'ownership',
        # Specific workplace scenarios
        'production issue', 'incident', 'outage', 'root cause', 'postmortem', 'trade-off',
        'tight deadline', 'deadlines', 'conflict', 'stakeholder',
        # Explicit hint
        'star', 'situation', 'task', 'action'
    ]
    return any(t in q for t in triggers)

def merge_results(primary: List[Dict], secondary: List[Dict], max_items: int = 10) -> List[Dict]:
    """Merge and dedupe retrieval results, keeping order and capping length."""
    out: List[Dict] = []
    seen = set()
    def add_item(item: Dict):
        key = item.get('id') or item.get('title')
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    for it in (primary or []):
        add_item(it)
    for it in (secondary or []):
        add_item(it)
    return out[:max_items]

def summarize_chunks(chunks: List[Dict], max_chars: int = 1200) -> str:
    """Simple extractive summarization by concatenating chunk excerpts until limit reached."""
    assembled: List[str] = []
    total = 0
    for ch in chunks:
        seg = f"[{ch.get('title')}] {ch.get('content')}"
        seg_len = len(seg)
        if total + seg_len > max_chars:
            remaining = max_chars - total
            if remaining > 60:  # avoid adding a too tiny fragment
                assembled.append(seg[:remaining])
            break
        assembled.append(seg)
        total += seg_len
    return "\n\n".join(assembled)

def answer_question(groq_client: Groq, index: Index, question: str, top_k: int = 10) -> str:
    """Retrieve top semantic matches, optionally summarize, then generate an answer."""
    if is_pii_request(question):
        return "I can discuss my professional experience and skills, but I don’t share personal contact or identification details."
    retrieved = semantic_search(index, question, top_k=top_k)
    if is_behavioral_query(question):
        star_results = semantic_search(index, question, top_k=6, tag='star')
        retrieved = merge_results(star_results, retrieved, max_items=top_k)
    if not retrieved:
        return "I don't have information related to that yet."
    long_context = len(retrieved) > 4 or any(len(r['content']) > 600 for r in retrieved)
    raw_context = summarize_chunks(retrieved) if long_context else "\n\n".join(f"[{r['title']}] {r['content']}" for r in retrieved)
    context_block = sanitize_text(raw_context)
    prompt = (
        "You are an AI digital twin responding in first person. Use only the provided context unless general knowledge is trivial.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer (first person):"
    )
    canonical_name = get_canonical_name()
    return generate_response_with_groq(groq_client, prompt, model=DEFAULT_MODEL, canonical_name=canonical_name)

def generate_response_with_groq(client, prompt, model=DEFAULT_MODEL, canonical_name: Optional[str] = None):
    """Generate response using Groq"""
    try:
        system_lines = [
            "You are an Tylor, have a strong background in IT, and industry knowledge in Hospitality. Answer questions as a person, speaking in first person about your background, skills, and experience.",
            "Do not provide personal contact or sensitive information (email, phone, address, DOB, IDs, bank details). If asked, politely decline and focus on professional information.",
            "Write concise, recruiter-ready answers. Prefer bullet points when listing items and include quantified outcomes when available.",
            "Keep answers between 50 and 200 words. Never include your thought process or internal reasoning in the response."
        ]
        if canonical_name:
            system_lines.append(
                f"The correct and canonical name is '{canonical_name}'. If the user uses any variant, nickname, or misspelling, always use the exact canonical spelling: '{canonical_name}'."
            )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role":"system","content":" ".join(system_lines)},
                {"role":"user","content":prompt}
            ],
            temperature=0.7,
            max_tokens=750
        )
        content = completion.choices[0].message.content.strip()
        # Normalize bullet formatting: ensure each bullet starts on its own line with '- '
        # Normalize bullets only at line starts; avoid mid-sentence replacements
        content = re.sub(r"(?m)^[ \t]*[•*\-]\s+", "- ", content)
        # Strip Markdown decorations like **bold**, *italic*, __bold__, _italic_, and inline code `x`
        content = re.sub(r"(\*\*|__)(.*?)\1", r"\2", content, flags=re.DOTALL)
        content = re.sub(r"(\*|_)(.*?)\1", r"\2", content, flags=re.DOTALL)
        content = re.sub(r"`([^`]*)`", r"\1", content)
        # Remove any stray asterisks left (not bullets which are now '- ')
        content = re.sub(r"\*+", "", content)
        # Remove decorative heading prefixes from each line
        content = "\n".join(_strip_known_prefixes(ln) for ln in content.splitlines())
        # Drop meta disclaimers / restated questions at the start
        content = clean_prologue(content)
        content = remove_standalone_headings(content)
        # If no bullets detected, convert to intro + bullets from sentences
        content = bulletize_if_needed(content)
        # Simplify bullet headings and enforce singular first-person
        content = simplify_bullet_headings(content)
        content = enforce_first_person_singular(content)
        content = unify_bullets(content)
        # Collapse multiple blank lines
        content = re.sub(r"\n{3,}", "\n\n", content)
        # Post trim to word limits and sentence boundary.
        content = limit_words(content, min_words=50, max_words=200)
        # Guarantee final punctuation.
        if not re.search(r'[.!?](?:[\'”"])?\s*$', content):
            content += "."
        return content
    except Exception as e:
        return f"❌ Error generating response: {e}"

def rag_query(index, groq_client, question):
    """Perform RAG query using Upstash Vector + Groq"""
    try:
        if is_pii_request(question):
            return "I can discuss my professional experience and skills, but I don’t share personal contact or identification details."
        results = semantic_search(index, question, top_k=8)
        if is_behavioral_query(question):
            star_results = semantic_search(index, question, top_k=6, tag='star')
            results = merge_results(star_results, results, max_items=8)
        if not results:
            return "I don't have specific information about that topic."
        print("🧠 Searching your professional profile...")
        top_docs = []
        for r in results:
            title = r.get('title','Information')
            content = r.get('content','')
            score = r.get('score',0.0)
            print(f"🔹 Found: {title} (Relevance: {score:.3f})")
            if content:
                top_docs.append(f"{title}: {content}")
        if not top_docs:
            return "I found some information but couldn't extract details."
        print("⚡ Generating personalized response...")
        context = sanitize_text("\n\n".join(top_docs))
        prompt = (
            "Based on the following information about yourself, answer the question.\n"
            "Speak in first person as describing your own background.\n\n"
            f"Your Information:\n{context}\n\nQuestion: {question}\n\nProvide a helpful, professional response:"
        )
        canonical_name = get_canonical_name()
        return generate_response_with_groq(groq_client, prompt, canonical_name=canonical_name)
    except Exception as e:
        return f"❌ Error during query: {e}"

def _load_all_chunks_from_profile() -> List[Dict]:
    """Load all chunks by rebuilding them from the pristine profile.

    We intentionally ignore any previously stored `content_chunks` key to avoid
    mutating or depending on JSON augmentation.
    """
    profile = load_profile_json(RESOLVED_JSON_FILE)
    if not profile:
        return []
    return build_content_chunks(profile) or []

def _first_sentence(text: str) -> str:
    if not text:
        return ""
    m = re.split(r"(?<=[.!?])\s+", text.strip())
    return m[0].strip() if m else text.strip()

def fallback_answer(index: Optional[Index], question: str, top_k: int = 8) -> str:
    """Heuristic answer when Groq or Vector DB is unavailable.

    - Prioritizes STAR/behavioral chunks if question is behavioral
    - Uses naive keyword overlap when no index is available
    - Produces intro + bullets, sanitized and word-limited
    """
    if is_pii_request(question):
        return "I can discuss my professional experience and skills, but I don’t share personal contact or identification details."
    results: List[Dict] = []
    if index is not None:
        results = semantic_search(index, question, top_k=top_k)
        if is_behavioral_query(question):
            star_results = semantic_search(index, question, top_k=6, tag='star')
            results = merge_results(star_results, results, max_items=top_k)
    else:
        chunks = _load_all_chunks_from_profile()
        if not chunks:
            return "I don't have information related to that yet."
        q_tokens = {t.lower() for t in re.findall(r"[a-zA-Z0-9]+", question)}
        scored: List[tuple[int, Dict]] = []
        for ch in chunks:
            text = f"{ch.get('title','')}. {ch.get('content','')}"
            tokens = {t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text)}
            score = len(q_tokens & tokens)
            # small boost for STAR
            if 'star' in (ch.get('metadata',{}).get('tags') or []):
                score += 2
            if score:
                scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, ch in scored[:top_k]:
            results.append({
                'title': ch.get('title'),
                'content': ch.get('content'),
                'score': 0.0,
                'category': ch.get('metadata',{}).get('category'),
                'tags': ch.get('metadata',{}).get('tags') or []
            })
    if not results:
        return "I don't have information related to that yet."
    # Build intro + bullets from top results
    intro = _strip_known_prefixes(_first_sentence(results[0].get('content','')))
    bullet_lines: List[str] = []
    for r in results[:5]:
        title = (r.get('title') or '').replace('Experience - ', '').strip()
        sent = _strip_known_prefixes(_first_sentence(r.get('content','')))
        bullet_lines.append(f"- {title}: {sent}")
    content = (intro + "\n\n" + "\n".join(bullet_lines)).strip()
    content = sanitize_text(content)
    content = clean_prologue(content)
    content = remove_standalone_headings(content)
    content = bulletize_if_needed(content)
    content = simplify_bullet_headings(content)
    content = enforce_first_person_singular(content)
    content = unify_bullets(content)
    content = limit_words(content, min_words=50, max_words=200)
    if not re.search(r'[.!?](?:[\'”"])?\s*$', content):
        content += "."
    return content

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Digital Twin RAG Assistant")
    parser.add_argument('--rebuild', action='store_true', help='(No-op) Embedding is managed by data/embed_digitaltwin.py')
    parser.add_argument('--show-count', action='store_true', help='Show current vector count then exit')
    parser.add_argument('--search', type=str, help='Run a semantic search over profile')
    parser.add_argument('--category', type=str, help='Optional category filter for --search')
    parser.add_argument('--tag', type=str, help='Optional tag filter for --search')
    parser.add_argument('--ask', type=str, help='Ask a one-off question and get an AI answer, then exit')
    return parser.parse_args()

def main():
    """Main application loop"""
    args = parse_args()
    print("🤖 Your Digital Twin - AI Profile Assistant")
    print("="*50)
    print("🔗 Vector Storage: Upstash (built-in embeddings)")
    print(f"⚡ AI Inference: Groq ({DEFAULT_MODEL})")
    print("📋 Data Source: Your Professional Profile")
    # Set up vector DB first (search/show-count do not require Groq)
    index = setup_vector_database(force_rebuild=args.rebuild)
    if not index:
        return
    if args.show_count:
        try:
            info = index.info()
            print(f"🔢 Vector count: {getattr(info,'vector_count',0)}")
        except Exception as e:
            print(f"⚠️ Unable to fetch vector count: {e}")
        return
    if args.search:
        results = semantic_search(index, args.search, top_k=8, category=args.category, tag=args.tag)
        if not results:
            print("No matches found.")
        else:
            print("🔍 Search Results:")
            for r in results:
                print(f"- {r['title']} (score={r['score']:.3f}, cat={r['category']}, tags={','.join(r['tags'])})")
        return
    if args.ask:
        groq_client = setup_groq_client()
        if not groq_client:
            return
        print(answer_question(groq_client, index, args.ask))
        return
    # Interactive chat requires Groq
    groq_client = setup_groq_client()
    if not groq_client:
        return
    print("✅ Your Digital Twin is ready!")
    print("🤖 Chat with your AI Digital Twin!")
    print("Ask questions about your experience, skills, projects, or career goals.")
    print("Type 'exit' to quit.")
    print("💭 Try asking:")
    print("  - 'Tell me about your work experience'")
    print("  - 'What are your technical skills?'")
    print("  - 'Describe your career goals'")
    print()
    while True:
        question = input("You: ")
        if question.lower() in ("exit","quit"):
            print("👋 Thanks for chatting with your Digital Twin!")
            break
        if question.strip():
            answer = rag_query(index, groq_client, question)
            print(f"🤖 Digital Twin: {answer}")

if __name__ == "__main__":
    main()