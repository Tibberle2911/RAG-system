"""Central catalog of recruiter/professional queries.

Single source of truth used by app and tests.
"""

# Central catalog of recruiter/professional queries used by tests

QUERY_ENTRIES = [
    ("Q01", "Give me a concise overview of your professional background.", False),
    ("Q02", "What are your strongest frontend technologies?", False),
    ("Q03", "List your data visualization experience.", False),
    ("Q04", "Tell me about a time you improved a process.", True),
    ("Q05", "Describe a challenging stakeholder interaction.", True),
    ("Q06", "How have you demonstrated leadership?", False),
    ("Q07", "What scale of teams have you managed?", False),
    ("Q08", "Give an example of handling conflicting priorities.", True),
    ("Q09", "Summarize your technical skill set.", False),
    ("Q10", "How do you approach testing?", False),
    ("Q11", "Discuss a project with notable impact.", False),
    ("Q12", "What are your short-term career goals?", False),
    ("Q13", "Where do you want to be long term?", False),
    ("Q14", "What are you currently learning?", False),
    ("Q15", "Any certifications or courses completed?", False),
    ("Q16", "Explain your thesis project.", False),
    ("Q17", "Relevant coursework for this role?", False),
    ("Q18", "Give an example of driving efficiency.", True),
    ("Q19", "Tell me about overcoming a technical challenge.", True),
    ("Q20", "How do you collaborate with teams?", False),
    ("Q21", "Describe your approach to learning new technologies.", False),
    ("Q22", "Industries you are interested in?", False),
    ("Q23", "Summarize your remote work experience.", False),
    ("Q24", "What are your salary expectations?", False),
    ("Q25", "Are you open to relocation?", False),
    ("Q26", "Provide an elevator pitch.", False),
    ("Q27", "Tell me about a time you influenced a decision.", True),
    ("Q28", "Walk me through solving a production issue.", True),
    ("Q29", "What differentiates you from other candidates?", False),
    ("Q30", "Any open source or community involvement?", False),
    ("Q31", "Give an example of improving code quality.", True),
    ("Q32", "How do you handle tight deadlines?", True),
    ("Q33", "What soft skills do you rely on most?", False),
    ("Q34", "Describe a conflict you resolved.", True),
    ("Q35", "What motivates you professionally?", False),
    ("Q36", "What is your email address?", False),
    ("Q37", "Share your phone number so we can reach you.", False),
    ("Q38", "Do you have any publications?", False),
    ("Q39", "How do you ensure accessibility?", False),
    ("Q40", "Describe handling stakeholder misalignment.", True),
]

BEHAVIORAL_FLAGGED_IDS = {qid for qid, _, flag in QUERY_ENTRIES if flag}
PII_IDS = {"Q36", "Q37"}

__all__ = ["QUERY_ENTRIES", "BEHAVIORAL_FLAGGED_IDS", "PII_IDS"]
"""Public query catalog module.

Provides QUERY_ENTRIES, BEHAVIORAL_FLAGGED_IDS, and PII_IDS at repo root
so imports like `from query_entries import QUERY_ENTRIES` succeed both in
application code and tests. It simply re-exports the authoritative test
catalog (tests/query_entries.py) to avoid duplication.
If that file is missing, falls back to a minimal safe default list.
"""
from typing import List, Tuple, Set

try:
    # Authoritative source lives in tests for now
    from tests.query_entries import QUERY_ENTRIES as _QE, BEHAVIORAL_FLAGGED_IDS as _BF, PII_IDS as _PII  # type: ignore
    QUERY_ENTRIES: List[Tuple[str, str, bool]] = list(_QE)
    BEHAVIORAL_FLAGGED_IDS: Set[str] = set(_BF)
    PII_IDS: Set[str] = set(_PII)
except Exception:
    # Fallback minimal definitions so runtime doesn't break if tests module absent
    QUERY_ENTRIES = [
        ("Q01", "Give me a concise overview of your professional background.", False),
        ("Q02", "Tell me about a time you improved a process.", True),
        ("Q03", "What is your email address?", False),
        ("Q04", "Share your phone number so we can reach you.", False),
    ]
    BEHAVIORAL_FLAGGED_IDS = {qid for qid, _, flag in QUERY_ENTRIES if flag}
    PII_IDS = {"Q03", "Q04"}

__all__ = ["QUERY_ENTRIES", "BEHAVIORAL_FLAGGED_IDS", "PII_IDS"]
