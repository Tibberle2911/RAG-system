import argparse
import sys
import os
import json
from typing import List, Dict

from upstash_vector import Index

# Reuse existing helpers and JSON path from the main RAG module
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root to path
import digitaltwin_rag as rag  # noqa: E402


def ensure_upstash() -> Index:
	if not rag.validate_upstash_env():
		raise SystemExit("Missing Upstash env vars. Add to .env: UPSTASH_VECTOR_REST_URL, UPSTASH_VECTOR_REST_TOKEN")
	return Index.from_env()


def build_vectors_from_json() -> List[tuple]:
	"""Profile-derived import without requiring `content_chunks`.

	- Uses rag.build_content_chunks(profile) to map the JSON into chunks
	- Excludes any dedicated STAR chunks (title contains 'STAR', case-insensitive)
	- Keeps everything else (e.g., Experience - Company, Technical Skills, etc.)
	"""
	profile = rag.load_profile_json(rag.RESOLVED_JSON_FILE)
	if not profile:
		raise SystemExit(f"Profile JSON not found or invalid at: {rag.RESOLVED_JSON_FILE}")
	chunks = rag.build_content_chunks(profile) or []
	# Optionally include structured behavioral stories from data/star_profile.json if present
	star_json_path = os.path.join(os.path.dirname(__file__), 'star_profile.json')
	if os.path.exists(star_json_path):
		try:
			with open(star_json_path, 'r', encoding='utf-8') as f:
				star_data: Dict = json.load(f)
			stories = star_data.get('stories', []) or []
			story_chunks: List[Dict] = []
			for i, s in enumerate(stories, start=1):
				company = s.get('company', '').strip()
				role = s.get('role', '').strip()
				title = f"Behavioral Story - {company} - {role}".strip(' -')
				content_lines = []
				if s.get('context'):
					content_lines.append(f"Context: {s.get('context')}")
				if s.get('situation'):
					content_lines.append(f"Situation: {s.get('situation')}")
				if s.get('task'):
					content_lines.append(f"Task: {s.get('task')}")
				if s.get('action'):
					content_lines.append(f"Action: {s.get('action')}")
				if s.get('result'):
					content_lines.append(f"Result: {s.get('result')}")
				story_chunks.append({
					'id': f'story_{i}',
					'title': title,
					'content': '\n'.join(content_lines),
					'type': 'experience',
					'metadata': {
						'category': 'experience',
						'tags': ['behavioral', 'methodology', 'star', company, role]
					}
				})
			if story_chunks:
				chunks.extend(story_chunks)
		except Exception as e:
			print(f"Warning: Unable to load star_profile.json: {e}")
	chunks = [c for c in chunks if 'star' not in str(c.get('title','')).lower()]
	vectors = rag.prepare_vectors_from_chunks(chunks)
	return vectors


def main():
	parser = argparse.ArgumentParser(description="Import profile data from JSON into Upstash Vector (no explicit content_chunks required)")
	parser.add_argument("--clear", action="store_true", help="Reset the index before import (destructive)")
	parser.add_argument("--show-count", action="store_true", help="Show index vector count and exit")
	# No flag to include STAR chunks; per requirement we always exclude them
	args = parser.parse_args()

	index = ensure_upstash()

	if args.show_count:
		try:
			info = index.info()
			print(f"Current vector count: {getattr(info, 'vector_count', 0)}")
		except Exception as e:
			print(f"Unable to fetch index info: {e}")
		return

	if args.clear:
		try:
			print("Resetting index (clearing all vectors)...")
			index.reset()
			print("Index reset complete.")
		except Exception as e:
			print(f"Failed to reset index: {e}")

	vectors = build_vectors_from_json()
	print(f"Upserting {len(vectors)} vectors to Upstash...")
	index.upsert(vectors=vectors)
	print("✅ Import complete.")


if __name__ == "__main__":
	main()

