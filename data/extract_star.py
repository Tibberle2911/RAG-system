import json
import os
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROFILE_PATH = os.path.join(BASE_DIR, 'data', 'digitaltwin.json')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'star_profile.json')


def load_profile(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_star(profile: Dict) -> List[Dict]:
    out: List[Dict] = []
    for exp in profile.get('experience', []):
        company = exp.get('company')
        title = exp.get('title')
        duration = exp.get('duration')
        context = exp.get('company_context')
        for story in exp.get('achievements_star', []):
            out.append({
                'company': company,
                'role': title,
                'duration': duration,
                'context': context,
                'situation': story.get('situation'),
                'task': story.get('task'),
                'action': story.get('action'),
                'result': story.get('result')
            })
    return out


def write_output(stories: List[Dict], path: str) -> None:
    data = {
        'star_methodology': 'Structured situation-task-action-result stories without STAR-titled chunks.',
        'count': len(stories),
        'stories': stories
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    if not os.path.exists(PROFILE_PATH):
        raise SystemExit(f'Profile JSON not found at {PROFILE_PATH}')
    profile = load_profile(PROFILE_PATH)
    stories = extract_star(profile)
    write_output(stories, OUTPUT_PATH)
    print(f'Extracted {len(stories)} STAR stories -> {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
