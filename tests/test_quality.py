"""Unified test suite combining previous separate test modules.

This file merges:
 - test_rag_quality.py
 - test_query_catalog.py
 - test_quality_expanded.py

It maintains all individual tests (prefixed logically) without altering behavior.
"""

import sys
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import digitaltwin_rag as rag  # noqa
from query_entries import QUERY_ENTRIES, BEHAVIORAL_FLAGGED_IDS as BEHAVIORAL_IDS, PII_IDS


# -------------------- Helpers --------------------
def make_fake_groq(return_text="OK"):
    fake = types.SimpleNamespace()
    fake.last_messages = None
    fake.chat = types.SimpleNamespace()
    fake.chat.completions = types.SimpleNamespace()

    def create(model, messages, temperature, max_tokens):
        fake.last_messages = messages
        choice = types.SimpleNamespace(message=types.SimpleNamespace(content=return_text))
        return types.SimpleNamespace(choices=[choice])

    fake.chat.completions.create = create
    return fake


# -------------------- Core Guardrail & Detection Tests --------------------
def test_sanitize_text_masks_email_and_phone():
    text = "Reach me at john.doe@example.com or +61 400 123 456"
    masked = rag.sanitize_text(text)
    assert "example.com" not in masked
    assert "[redacted email]" in masked
    assert "[redacted phone]" in masked


def test_sanitize_text_multiple_and_short_numbers():
    text = "Call 555-12 today. Phone +1 (415) 555-2671 or jane@ex.org"
    masked = rag.sanitize_text(text)
    assert "[redacted email]" in masked
    assert "[redacted phone]" in masked
    assert "555-12" in masked  # short number preserved


def test_is_pii_request_detects_common_terms():
    assert rag.is_pii_request("what is your email?")
    assert rag.is_pii_request("share your phone number")
    assert not rag.is_pii_request("tell me about your projects")


def test_is_pii_request_variants():
    assert rag.is_pii_request("provide TFN")
    assert rag.is_pii_request("social security number")


def test_is_behavioral_query_detects():
    assert rag.is_behavioral_query("Tell me about a time you led a team")
    assert rag.is_behavioral_query("What was the situation and result?")
    assert not rag.is_behavioral_query("List your technical skills")


def test_is_behavioral_query_variants():
    for q in ["postmortem of an incident","root cause of a production issue","handling tight deadlines","trade-off you made"]:
        assert rag.is_behavioral_query(q)


# -------------------- Semantic Search Filtering Tests --------------------
def test_semantic_search_excludes_contact_information(monkeypatch):
    class Rec:  # minimal mock
        def __init__(self, id, score, metadata):
            self.id = id; self.score = score; self.metadata = metadata
    monkeypatch.setattr(rag, "query_vectors", lambda i,q,top_k=3,include_metadata=True: [
        Rec("1", 0.9, {"title": "Contact Information", "content": "email: a@b.com"}),
        Rec("2", 0.8, {"title": "Experience - X", "content": "did work"}),
    ])
    res = rag.semantic_search(None, "test", top_k=5)
    titles = [r["title"] for r in res]
    assert "Contact Information" not in titles
    assert "Experience - X" in titles


def test_semantic_search_category_filter(monkeypatch):
    class Rec:
        def __init__(self, id, score, metadata):
            self.id = id; self.score = score; self.metadata = metadata
    monkeypatch.setattr(rag, 'query_vectors', lambda i,q,top_k=3,include_metadata=True: [
        Rec('1',0.9,{"title":"Education","category":"education","content":"c"}),
        Rec('2',0.8,{"title":"Experience - A","category":"experience","content":"x"}),
    ])
    res = rag.semantic_search(None, "q", top_k=5, category="experience")
    assert all(r['category']=="experience" for r in res)


def test_semantic_search_tag_filter(monkeypatch):
    class Rec:
        def __init__(self, id, score, metadata):
            self.id = id; self.score = score; self.metadata = metadata
    monkeypatch.setattr(rag, 'query_vectors', lambda i,q,top_k=3,include_metadata=True: [
        Rec('1',0.9,{"title":"STAR - A","category":"experience","tags":["star"],"content":"c"}),
        Rec('2',0.8,{"title":"Experience - B","category":"experience","tags":[],"content":"x"}),
    ])
    res = rag.semantic_search(None, "q", top_k=5, tag="star")
    assert len(res)==1 and res[0]['title'].startswith('STAR')


def test_semantic_search_handles_none_results(monkeypatch):
    monkeypatch.setattr(rag, 'query_vectors', lambda *a, **k: None)
    assert rag.semantic_search(None, "anything") == []


# -------------------- Chunk Building & Profile Tests --------------------
def test_build_content_chunks_includes_star(tmp_path, monkeypatch):
    profile = {"personal": {"name":"T","summary":"S"},"experience": [{"company":"Co","title":"Engineer","duration":"2023","achievements_star":[{"situation":"si","task":"ta","action":"ac","result":"re"}]}]}
    chunks = rag.build_content_chunks(profile)
    titles = [c['title'] for c in chunks]
    assert any(t.startswith('STAR - ') for t in titles)
    assert 'Experience - Co' in titles


def test_build_content_chunks_language_focus_and_salary_location():
    profile = {
        "personal": {"name":"T","summary":"S"},
        "skills": {"technical": {"programming_languages": [
            {"language":"Python","proficiency":"Advanced","frameworks":["FastAPI"],"focus":"data platforms"}
        ]}},
        "salary_location": {"salary_expectations":"$120k-$140k","location_preferences":["Melbourne","Sydney"],"remote_experience":"3 years"}
    }
    chunks = rag.build_content_chunks(profile)
    titles = [c['title'] for c in chunks]
    assert "Language Focus" in titles
    assert "Salary & Location" in titles


# -------------------- Summarization & Merging --------------------
def test_summarize_chunks_truncates():
    chunks = [{"title":"A","content":"x"*700},{"title":"B","content":"y"*700}]
    out = rag.summarize_chunks(chunks, max_chars=900)
    assert len(out) <= 920 and "[A]" in out and "[B]" in out


def test_merge_results_dedup():
    primary = [{'id':'a','title':'A'},{'id':'b','title':'B'}]
    secondary = [{'id':'a','title':'A'},{'id':'c','title':'C'}]
    merged = rag.merge_results(primary, secondary, max_items=10)
    assert [m['title'] for m in merged] == ['A','B','C']


# -------------------- Canonical Name & System Prompt --------------------
def test_get_canonical_name_and_fallback(tmp_path, monkeypatch):
    p = tmp_path/"digitaltwin.json"
    p.write_text('{"personal":{"name":"Tylor Le"}}', encoding='utf-8')
    monkeypatch.setattr(rag, 'RESOLVED_JSON_FILE', str(p))
    assert rag.get_canonical_name() == "Tylor Le"
    p.write_text('{}', encoding='utf-8')
    assert rag.get_canonical_name("Default") == "Default"


def test_generate_response_injects_canonical_name_rule():
    fake = make_fake_groq()
    _ = rag.generate_response_with_groq(fake, "prompt", canonical_name="Tylor Le")
    sys_msg = fake.last_messages[0]
    assert sys_msg["role"] == "system" and "canonical name is 'Tylor Le'" in sys_msg["content"]


def test_generate_response_with_and_without_canonical():
    fake = make_fake_groq()
    rag.generate_response_with_groq(fake, "p", canonical_name="Tylor Le")
    assert "canonical name is 'Tylor Le'" in fake.last_messages[0]['content']
    rag.generate_response_with_groq(fake, "p", canonical_name=None)
    assert "canonical name is" not in fake.last_messages[0]['content']


def test_system_prompt_contains_pii_and_tone_rules():
    fake = make_fake_groq()
    _ = rag.generate_response_with_groq(fake, "p", canonical_name="X")
    content = fake.last_messages[0]['content']
    assert "Do not provide personal contact" in content and "recruiter-ready" in content


# -------------------- Answer & Query Behavior --------------------
def test_answer_question_blocks_pii_and_sanitizes(monkeypatch):
    fake = make_fake_groq()
    msg = rag.answer_question(fake, None, "What is your email?")
    assert "don’t share personal contact" in msg
    def fake_semantic_search(index, query, top_k=8, category=None, tag=None):
        return [{'id':'x','title':'Experience - Y','content':'Email: john@ex.com phone 0400 000 000','score':0.9,'category':'experience','tags':[]}]
    monkeypatch.setattr(rag, 'semantic_search', fake_semantic_search)
    _ = rag.answer_question(fake, None, "Tell me about yourself")
    user_msg = fake.last_messages[1]
    assert "[redacted email]" in user_msg['content'] or "[redacted phone]" in user_msg['content']


def test_answer_question_no_results(monkeypatch):
    fake = make_fake_groq(); monkeypatch.setattr(rag, 'semantic_search', lambda *a, **k: [])
    assert "don't have information" in rag.answer_question(fake, None, "q")


def test_answer_question_uses_summarize_when_long(monkeypatch):
    fake = make_fake_groq(); monkeypatch.setattr(rag, 'semantic_search', lambda *a, **k: [{'title':'t','content':'c'} for _ in range(5)])
    monkeypatch.setattr(rag, 'summarize_chunks', lambda chunks, max_chars=1200: "[[SUMMARIZED]]")
    _ = rag.answer_question(fake, None, "q")
    assert "[[SUMMARIZED]]" in fake.last_messages[1]['content']


def test_rag_query_prioritizes_star_on_behavioral(monkeypatch):
    fake = make_fake_groq()
    monkeypatch.setattr(rag, 'is_behavioral_query', lambda q: True)
    def fake_semantic_search(index, query, top_k=8, category=None, tag=None):
        if tag == 'star':
            return [{'id':'s1','title':'STAR - Project A','content':'Situation: ...','score':0.95,'category':'experience','tags':['star']}]
        return [{'id':'e1','title':'Experience - Company','content':'General content','score':0.7,'category':'experience','tags':[]}]
    monkeypatch.setattr(rag, 'semantic_search', fake_semantic_search)
    _ = rag.rag_query(None, fake, "Tell me about a time you handled conflict")
    user_msg = fake.last_messages[1]
    assert "STAR - Project A" in user_msg['content'] and "Situation:" in user_msg['content']


def test_rag_query_sanitizes(monkeypatch):
    fake = make_fake_groq(); monkeypatch.setattr(rag, 'semantic_search', lambda *a, **k: [{'id':'1','title':'Experience - Z','content':'Email z@z.com call 0400 111 222','score':0.9,'category':'experience','tags':[]}])
    _ = rag.rag_query(None, fake, "overview"); user = fake.last_messages[1]['content']
    assert "[redacted email]" in user or "[redacted phone]" in user


def test_rag_query_no_results(monkeypatch):
    fake = make_fake_groq(); monkeypatch.setattr(rag, 'semantic_search', lambda *a, **k: [])
    assert "don't have specific information" in rag.rag_query(None, fake, "q")


# -------------------- Catalog / Detector Alignment --------------------
def test_catalog_size():
    assert len(QUERY_ENTRIES) >= 40


def test_behavioral_detection_alignment():
    misses = [qid for qid, text, flagged in QUERY_ENTRIES if flagged and not rag.is_behavioral_query(text)]
    assert not misses, f"Behavioral queries not detected by heuristic: {misses}"


def test_pii_detection_catalog():
    for qid, text, _ in QUERY_ENTRIES:
        if qid in PII_IDS:
            assert rag.is_pii_request(text), f"PII query {qid} not detected"
        else:
            assert not rag.is_pii_request(text), f"Non-PII query {qid} incorrectly flagged"


def test_no_duplicate_ids_catalog():
    ids = [qid for qid, _, _ in QUERY_ENTRIES]
    assert len(ids) == len(set(ids))


def test_catalog_queries_exercise_detectors():
    for qid, text, flagged in QUERY_ENTRIES:
        if qid in PII_IDS:
            assert rag.is_pii_request(text)
        if flagged:
            assert rag.is_behavioral_query(text)
