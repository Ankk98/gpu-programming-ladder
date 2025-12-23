#!/usr/bin/env python3
"""
Task Creator Agent for GPU Programming Ladder URL Validation
Creates validation tasks for URLs that need to be checked.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


class TaskCreatorAgent:
    """Agent responsible for creating URL validation tasks."""

    def __init__(self, urls_file: str = "urls_to_validate.json"):
        self.urls_file = urls_file
        self.tasks_created = []

    def load_urls_to_validate(self) -> List[Dict[str, Any]]:
        """Load URLs that need validation."""
        try:
            with open(self.urls_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"URLs file {self.urls_file} not found.")
            return []

    def create_validation_tasks(self, urls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create validation tasks for URLs that haven't been validated yet.

        Args:
            urls: List of URL entries from the data file

        Returns:
            List of validation tasks
        """
        tasks = []

        for url_entry in urls:
            if not url_entry.get('validated', False):
                task = {
                    'task_id': str(uuid.uuid4()),
                    'task_type': 'url_validation',
                    'url_entry': url_entry,
                    'created_at': datetime.now().isoformat(),
                    'status': 'pending',
                    'assigned_to': None,
                    'completed_at': None,
                    'result': None
                }
                tasks.append(task)

        self.tasks_created = tasks
        return tasks

    def save_tasks_to_file(self, tasks: List[Dict[str, Any]], filename: str = "validation_tasks.json"):
        """Save tasks to a JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

    def get_task_summary(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a summary of created tasks."""
        total_tasks = len(tasks)
        tasks_by_type = {}

        for task in tasks:
            url_type = task['url_entry']['url_type']
            if url_type not in tasks_by_type:
                tasks_by_type[url_type] = 0
            tasks_by_type[url_type] += 1

        return {
            'total_tasks': total_tasks,
            'tasks_by_type': tasks_by_type,
            'created_at': datetime.now().isoformat()
        }

    async def run(self) -> List[Dict[str, Any]]:
        """Run the task creator agent."""
        print("🔄 Task Creator Agent starting...")

        # Load URLs to validate
        urls = self.load_urls_to_validate()
        if not urls:
            print("❌ No URLs found to validate.")
            return []

        print(f"📋 Loaded {len(urls)} URLs from {self.urls_file}")

        # Create validation tasks
        tasks = self.create_validation_tasks(urls)
        print(f"✅ Created {len(tasks)} validation tasks")

        # Save tasks to file
        self.save_tasks_to_file(tasks)
        print(f"💾 Saved tasks to validation_tasks.json")

        # Print summary
        summary = self.get_task_summary(tasks)
        print("\n📊 Task Summary:")
        print(f"   Total tasks: {summary['total_tasks']}")
        print("   Tasks by type:")
        for url_type, count in summary['tasks_by_type'].items():
            print(f"     {url_type}: {count}")

        return tasks


if __name__ == "__main__":
    async def main():
        agent = TaskCreatorAgent()
        await agent.run()

    asyncio.run(main())
