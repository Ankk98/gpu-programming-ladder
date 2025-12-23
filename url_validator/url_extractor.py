#!/usr/bin/env python3
"""
URL Extractor for GPU Programming Ladder data.js
Extracts all URLs from the ladder data structure for validation.
"""

import json5
import json
import re
from typing import List, Dict, Any
from pathlib import Path


def extract_urls_from_data_js(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract all URLs from the data.js file.

    Args:
        file_path: Path to the data.js file

    Returns:
        List of URL entries with metadata for validation
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract the array content using regex
    match = re.search(r'const ladder = (\[[\s\S]*?\]);', content)
    if not match:
        raise ValueError("Could not find ladder array in data.js")

    json_content = match.group(1)

    # Parse the JSON5 content
    ladder_data = json5.loads(json_content)

    urls_to_validate = []

    for phase in ladder_data:
        phase_id = phase['id']
        phase_name = phase['name']

        for group in phase.get('groups', []):
            group_id = group['id']
            group_title = group['title']

            for topic in group.get('topics', []):
                topic_id = topic['id']
                topic_title = topic['title']

                # Define URL fields to check
                url_fields = ['article', 'paper', 'video', 'exercise', 'python', 'cpp']

                for field in url_fields:
                    if field in topic and topic[field]:
                        url_entry = {
                            'phase_id': phase_id,
                            'phase_name': phase_name,
                            'group_id': group_id,
                            'group_title': group_title,
                            'topic_id': topic_id,
                            'topic_title': topic_title,
                            'url_type': field,
                            'url': topic[field],
                            'validated': False,
                            'is_valid': None,
                            'replacement_url': None,
                            'validation_notes': None
                        }
                        urls_to_validate.append(url_entry)

    return urls_to_validate


def save_urls_to_json(urls: List[Dict[str, Any]], output_file: str):
    """Save extracted URLs to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(urls, f, indent=2, ensure_ascii=False)


def load_urls_from_json(input_file: str) -> List[Dict[str, Any]]:
    """Load URLs from a JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    # Extract URLs from data.js
    urls = extract_urls_from_data_js('data.js')

    print(f"Extracted {len(urls)} URLs from data.js")

    # Group by URL type for summary
    url_types = {}
    for url in urls:
        url_type = url['url_type']
        if url_type not in url_types:
            url_types[url_type] = 0
        url_types[url_type] += 1

    print("\nURL types distribution:")
    for url_type, count in url_types.items():
        print(f"  {url_type}: {count}")

    # Save to JSON file
    save_urls_to_json(urls, 'urls_to_validate.json')
    print("\nSaved URLs to urls_to_validate.json")
