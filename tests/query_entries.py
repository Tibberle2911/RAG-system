"""Test helper module re-exporting the single authoritative catalog.

This file intentionally avoids duplicating data. The root-level
`query_entries.py` holds the catalog; tests import from here for clarity.
"""

from query_entries import QUERY_ENTRIES, BEHAVIORAL_FLAGGED_IDS, PII_IDS  # noqa: F401

__all__ = ["QUERY_ENTRIES", "BEHAVIORAL_FLAGGED_IDS", "PII_IDS"]
