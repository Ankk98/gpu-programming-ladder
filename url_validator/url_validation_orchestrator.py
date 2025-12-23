#!/usr/bin/env python3
"""
URL Validation Orchestrator for GPU Programming Ladder
Coordinates task creation and consumer agents for URL validation.
"""

import json
import asyncio
import aiofiles
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor
import threading

from task_creator_agent import TaskCreatorAgent
from url_validator_agent import URLValidatorAgent


class URLValidationOrchestrator:
    """Orchestrates the URL validation process with multiple agents."""

    def __init__(self,
                 num_consumer_agents: int = 3,
                 max_concurrent_requests_per_agent: int = 5,
                 lm_studio_url: str = "http://localhost:1234",
                 tasks_file: str = "validation_tasks.json",
                 results_file: str = "validation_results.json"):
        self.num_consumer_agents = num_consumer_agents
        self.max_concurrent_requests_per_agent = max_concurrent_requests_per_agent
        self.lm_studio_url = lm_studio_url
        self.tasks_file = tasks_file
        self.results_file = results_file

        # Thread safety
        self.lock = threading.Lock()
        self.completed_tasks = []
        self.agent_stats = {}

    async def run_validation_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete URL validation pipeline.

        Returns:
            Summary of validation results
        """
        print("🚀 Starting URL Validation Pipeline")
        print(f"   Consumer agents: {self.num_consumer_agents}")
        print(f"   Max concurrent requests per agent: {self.max_concurrent_requests_per_agent}")
        print(f"   LM Studio URL: {self.lm_studio_url}")

        # Step 1: Create tasks
        print("\n📋 Step 1: Creating validation tasks...")
        task_creator = TaskCreatorAgent()
        tasks = await task_creator.run()

        if not tasks:
            return {"error": "No tasks created"}

        # Step 2: Distribute tasks to consumer agents
        print("\n🤖 Step 2: Starting consumer agents...")
        await self._run_consumer_agents(tasks)

        # Step 3: Save results
        print("\n💾 Step 3: Saving validation results...")
        await self._save_results()

        # Step 4: Generate summary
        summary = self._generate_summary()
        print("\n📊 Step 4: Validation complete!")
        self._print_summary(summary)

        return summary

    async def _run_consumer_agents(self, tasks: List[Dict[str, Any]]):
        """Run consumer agents to validate URLs."""
        # Split tasks among agents
        tasks_per_agent = len(tasks) // self.num_consumer_agents
        remainder = len(tasks) % self.num_consumer_agents

        task_batches = []
        start_idx = 0

        for i in range(self.num_consumer_agents):
            # Distribute remainder tasks to first few agents
            batch_size = tasks_per_agent + (1 if i < remainder else 0)
            end_idx = start_idx + batch_size
            task_batches.append(tasks[start_idx:end_idx])
            start_idx = end_idx

        # Run agents concurrently
        agent_coroutines = []
        for i, task_batch in enumerate(task_batches):
            agent_id = f"agent_{i+1}"
            coroutine = self._run_single_agent(agent_id, task_batch)
            agent_coroutines.append(coroutine)

        await asyncio.gather(*agent_coroutines)

    async def _run_single_agent(self, agent_id: str, tasks: List[Dict[str, Any]]):
        """Run a single consumer agent."""
        print(f"   🟢 Agent {agent_id} starting with {len(tasks)} tasks")

        async with URLValidatorAgent(
            agent_id=agent_id,
            lm_studio_url=self.lm_studio_url,
            max_concurrent_requests=self.max_concurrent_requests_per_agent
        ) as agent:

            processed_tasks = await agent.process_tasks(tasks)

            # Thread-safe update of completed tasks
            with self.lock:
                self.completed_tasks.extend(processed_tasks)
                self.agent_stats[agent_id] = {
                    'tasks_processed': len(processed_tasks),
                    'total_urls': len(tasks)
                }

        print(f"   ✅ Agent {agent_id} completed {len(processed_tasks)} tasks")

    async def _save_results(self):
        """Save validation results to file."""
        # Prepare results data
        results_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'num_agents': self.num_consumer_agents,
                'max_concurrent_requests_per_agent': self.max_concurrent_requests_per_agent,
                'lm_studio_url': self.lm_studio_url,
                'agent_stats': self.agent_stats
            },
            'results': []
        }

        # Extract URL validation results
        for task in self.completed_tasks:
            if task.get('result'):
                results_data['results'].append(task['result'])

        # Save to file
        async with aiofiles.open(self.results_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(results_data, indent=2, ensure_ascii=False))

        print(f"   💾 Saved {len(results_data['results'])} results to {self.results_file}")

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of validation results."""
        if not self.completed_tasks:
            return {"error": "No tasks completed"}

        total_urls = len(self.completed_tasks)
        valid_urls = 0
        invalid_urls = 0
        replaced_urls = 0
        removed_urls = 0

        urls_by_type = {}
        replacements_by_type = {}

        for task in self.completed_tasks:
            result = task.get('result', {})
            url_type = result.get('url_type', 'unknown')
            is_valid = result.get('is_valid', False)
            has_replacement = result.get('replacement_url') is not None

            if url_type not in urls_by_type:
                urls_by_type[url_type] = {'total': 0, 'valid': 0, 'invalid': 0, 'replaced': 0}

            urls_by_type[url_type]['total'] += 1

            if is_valid:
                valid_urls += 1
                urls_by_type[url_type]['valid'] += 1
            else:
                invalid_urls += 1
                urls_by_type[url_type]['invalid'] += 1

                if has_replacement:
                    replaced_urls += 1
                    urls_by_type[url_type]['replaced'] += 1
                else:
                    removed_urls += 1

        return {
            'total_urls_processed': total_urls,
            'valid_urls': valid_urls,
            'invalid_urls': invalid_urls,
            'replaced_urls': replaced_urls,
            'removed_urls': removed_urls,
            'urls_by_type': urls_by_type,
            'success_rate': (valid_urls + replaced_urls) / total_urls * 100 if total_urls > 0 else 0,
            'generated_at': datetime.now().isoformat()
        }

    def _print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary."""
        print("   📊 Validation Summary:")
        print(f"      Total URLs processed: {summary['total_urls_processed']}")
        print(f"      Valid URLs: {summary['valid_urls']}")
        print(f"      Invalid URLs: {summary['invalid_urls']}")
        print(f"      URLs with replacements: {summary['replaced_urls']}")
        print(f"      URLs to be removed: {summary['removed_urls']}")
        print(f"      Success rate: {summary['success_rate']:.1f}%")
        print("\n   Breakdown by URL type:")
        for url_type, stats in summary['urls_by_type'].items():
            total = stats['total']
            valid = stats['valid']
            invalid = stats['invalid']
            replaced = stats['replaced']
            print(f"      {url_type}: {valid}/{total} valid, {replaced} replaced, {invalid-replaced} to remove")

    async def update_data_js_with_results(self, data_js_file: str = "data.js") -> bool:
        """
        Update the original data.js file with validation results.

        Args:
            data_js_file: Path to the data.js file to update

        Returns:
            True if update was successful
        """
        try:
            # Load current results
            async with aiofiles.open(self.results_file, 'r', encoding='utf-8') as f:
                results_data = json.loads(await f.read())

            # Load original data.js
            async with aiofiles.open(data_js_file, 'r', encoding='utf-8') as f:
                original_content = await f.read()

            # Parse the ladder data
            import json5
            import re

            match = re.search(r'const ladder = (\[[\s\S]*?\]);', original_content)
            if not match:
                print("❌ Could not parse data.js structure")
                return False

            ladder_data = json5.loads(match.group(1))

            # Create mapping of results by topic and URL type
            results_map = {}
            for result in results_data['results']:
                key = f"{result['topic_id']}_{result['url_type']}"
                results_map[key] = result

            # Update ladder data with results
            urls_updated = 0
            urls_removed = 0

            for phase in ladder_data:
                for group in phase.get('groups', []):
                    for topic in group.get('topics', []):
                        for url_type in ['article', 'paper', 'video', 'exercise', 'python', 'cpp']:
                            if url_type in topic:
                                key = f"{topic['id']}_{url_type}"
                                if key in results_map:
                                    result = results_map[key]

                                    if result.get('replacement_url'):
                                        # Replace URL
                                        topic[url_type] = result['replacement_url']
                                        urls_updated += 1
                                        print(f"🔄 Updated {url_type} for {topic['title'][:50]}...")
                                    elif not result.get('is_valid', True):
                                        # Remove invalid URL
                                        del topic[url_type]
                                        urls_removed += 1
                                        print(f"🗑️  Removed invalid {url_type} for {topic['title'][:50]}...")

            # Write updated data back to file
            updated_json = json.dumps(ladder_data, indent=2, ensure_ascii=False)
            updated_content = original_content.replace(match.group(1), updated_json)

            async with aiofiles.open(data_js_file, 'w', encoding='utf-8') as f:
                await f.write(updated_content)

            print(f"✅ Updated data.js: {urls_updated} URLs replaced, {urls_removed} URLs removed")
            return True

        except Exception as e:
            print(f"❌ Error updating data.js: {str(e)}")
            return False
