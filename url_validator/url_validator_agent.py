#!/usr/bin/env python3
"""
URL Validator Agent for GPU Programming Ladder
Validates URLs and finds replacements when needed.
"""

import json
import asyncio
import aiohttp
import aiofiles
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import os


class URLValidatorAgent:
    """Agent responsible for validating URLs and finding replacements."""

    def __init__(self,
                 agent_id: str,
                 lm_studio_url: str = "http://localhost:1234",
                 max_concurrent_requests: int = 5,
                 timeout: int = 30):
        self.agent_id = agent_id
        self.lm_studio_url = lm_studio_url
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def check_url_exists(self, url: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a URL exists and is accessible.

        Args:
            url: URL to check

        Returns:
            Tuple of (exists: bool, error_message: Optional[str])
        """
        if not url or not url.startswith(('http://', 'https://')):
            return False, "Invalid URL format"

        try:
            async with self.session.get(url, allow_redirects=True) as response:
                # Consider 2xx and 3xx status codes as valid (redirects are OK)
                if response.status < 400:
                    return True, None
                else:
                    return False, f"HTTP {response.status}: {response.reason}"
        except aiohttp.ClientError as e:
            return False, f"Network error: {str(e)}"
        except asyncio.TimeoutError:
            return False, "Request timeout"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    async def analyze_url_content(self, url: str, url_type: str) -> Dict[str, Any]:
        """
        Analyze URL content to determine if it's appropriate for the URL type.

        Args:
            url: URL to analyze
            url_type: Type of URL (article, video, exercise, etc.)

        Returns:
            Dictionary with analysis results
        """
        try:
            async with self.session.get(url) as response:
                if response.status >= 400:
                    return {
                        'is_appropriate': False,
                        'reason': f"HTTP {response.status}",
                        'content_type': None,
                        'title': None
                    }

                content_type = response.headers.get('content-type', '').lower()
                html_content = await response.text()

                # Parse HTML if it's HTML content
                if 'text/html' in content_type:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    title = soup.title.string.strip() if soup.title else None

                    # Check if it's a listing page vs specific content page
                    analysis = self._analyze_html_content(url, url_type, soup, title)
                    analysis['content_type'] = content_type
                    analysis['title'] = title
                    return analysis
                else:
                    # Non-HTML content (PDF, etc.)
                    return {
                        'is_appropriate': True,  # Assume non-HTML is specific content
                        'reason': 'Non-HTML content (likely direct file)',
                        'content_type': content_type,
                        'title': None
                    }

        except Exception as e:
            return {
                'is_appropriate': False,
                'reason': f"Content analysis failed: {str(e)}",
                'content_type': None,
                'title': None
            }

    def _analyze_html_content(self, url: str, url_type: str, soup: BeautifulSoup, title: str) -> Dict[str, Any]:
        """Analyze HTML content for appropriateness."""
        # Check for common listing page indicators
        listing_indicators = [
            'list', 'index', 'directory', 'category', 'archive',
            'all challenges', 'all exercises', 'tutorials',
            'courses', 'curriculum'
        ]

        text_content = soup.get_text().lower()
        url_lower = url.lower()

        # Check if title suggests a listing
        title_listing = any(indicator in (title or '').lower() for indicator in listing_indicators)

        # Check if URL path suggests a listing
        url_listing = any(indicator in url_lower for indicator in listing_indicators)

        # Check if content has multiple links that might indicate a listing
        links = soup.find_all('a', href=True)
        many_links = len(links) > 20  # Arbitrary threshold

        # Specific checks based on URL type
        if url_type in ['exercise', 'video']:
            # For exercises and videos, we want specific content, not listings
            if title_listing or url_listing or many_links:
                return {
                    'is_appropriate': False,
                    'reason': 'Appears to be a listing/index page rather than specific content'
                }
        elif url_type == 'article':
            # Articles can be either specific or listing, but prefer specific
            if title_listing and many_links:
                return {
                    'is_appropriate': False,
                    'reason': 'Appears to be a general listing page'
                }

        # GitHub repository checks
        if 'github.com' in url:
            path_parts = urlparse(url).path.strip('/').split('/')
            if len(path_parts) == 2:  # Just org/repo
                return {
                    'is_appropriate': False,
                    'reason': 'GitHub repository root - should link to specific file or directory'
                }

        return {
            'is_appropriate': True,
            'reason': 'Content appears appropriate'
        }

    async def find_replacement_url(self, url_entry: Dict[str, Any]) -> Optional[str]:
        """
        Use LLM to find a replacement URL for broken/inappropriate URLs.

        Args:
            url_entry: URL entry with metadata

        Returns:
            Replacement URL or None if no good replacement found
        """
        prompt = f"""
You are an expert in GPU programming and machine learning education. I need you to find a replacement URL for the following broken or inappropriate link:

Topic: {url_entry['topic_title']}
Phase: {url_entry['phase_name']}
Group: {url_entry['group_title']}
URL Type: {url_entry['url_type']}
Original URL: {url_entry['url']}

The original URL is either broken or points to a listing page instead of specific content.
For {url_entry['url_type']} URLs, I need links that point to:
- article: Specific article/tutorial page
- paper: Direct paper PDF or abstract page
- video: Specific video (not playlist or channel)
- exercise: Specific exercise/challenge (not listing of all exercises)
- python/cpp: Specific code repository or documentation

Please find a high-quality replacement URL that matches this topic. Return only the URL, or "NO_REPLACEMENT" if you cannot find a suitable replacement.

Consider these domains for replacements:
- developer.nvidia.com (CUDA documentation)
- pytorch.org (PyTorch docs)
- github.com (code repositories)
- arxiv.org (papers)
- youtube.com (specific videos)
- coursera.org, edX.org (courses, but prefer specific lessons)
- towardsdatascience.com, medium.com (articles)
"""

        try:
            async with self.session.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "model": "gpt-4o-20b",  # Adjust based on your LM Studio model
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.1
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    replacement_url = result['choices'][0]['message']['content'].strip()

                    if replacement_url and replacement_url != "NO_REPLACEMENT":
                        # Validate the replacement URL
                        exists, error = await self.check_url_exists(replacement_url)
                        if exists:
                            return replacement_url
                        else:
                            print(f"⚠️  Replacement URL {replacement_url} is also invalid: {error}")
                            return None
                    else:
                        return None
                else:
                    print(f"❌ LLM API error: {response.status}")
                    return None

        except Exception as e:
            print(f"❌ Error calling LLM: {str(e)}")
            return None

    async def validate_url_entry(self, url_entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a single URL entry.

        Args:
            url_entry: URL entry to validate

        Returns:
            Updated URL entry with validation results
        """
        url = url_entry['url']
        url_type = url_entry['url_type']

        print(f"🔍 Validating {url_type} URL: {url}")

        # Check if URL exists
        exists, error = await self.check_url_exists(url)

        if not exists:
            print(f"❌ URL does not exist: {error}")
            # Try to find replacement
            replacement = await self.find_replacement_url(url_entry)
            if replacement:
                print(f"✅ Found replacement: {replacement}")
                return {
                    **url_entry,
                    'validated': True,
                    'is_valid': False,
                    'replacement_url': replacement,
                    'validation_notes': f"Original URL broken ({error}). Replaced with: {replacement}"
                }
            else:
                print("❌ No replacement found")
                return {
                    **url_entry,
                    'validated': True,
                    'is_valid': False,
                    'replacement_url': None,
                    'validation_notes': f"URL broken ({error}) and no suitable replacement found"
                }

        # URL exists, check if content is appropriate
        content_analysis = await self.analyze_url_content(url, url_type)

        if not content_analysis['is_appropriate']:
            print(f"⚠️  URL exists but content inappropriate: {content_analysis['reason']}")
            # Try to find replacement
            replacement = await self.find_replacement_url(url_entry)
            if replacement:
                print(f"✅ Found replacement: {replacement}")
                return {
                    **url_entry,
                    'validated': True,
                    'is_valid': False,
                    'replacement_url': replacement,
                    'validation_notes': f"URL exists but inappropriate ({content_analysis['reason']}). Replaced with: {replacement}"
                }
            else:
                print("❌ No replacement found")
                return {
                    **url_entry,
                    'validated': True,
                    'is_valid': False,
                    'replacement_url': None,
                    'validation_notes': f"URL exists but inappropriate ({content_analysis['reason']}) and no suitable replacement found"
                }

        # URL is valid and appropriate
        print(f"✅ URL is valid and appropriate")
        return {
            **url_entry,
            'validated': True,
            'is_valid': True,
            'replacement_url': None,
            'validation_notes': content_analysis['reason']
        }

    async def process_tasks(self, tasks: list) -> list:
        """
        Process a batch of validation tasks.

        Args:
            tasks: List of tasks to process

        Returns:
            List of processed tasks
        """
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        processed_tasks = []

        async def process_task_with_semaphore(task):
            async with semaphore:
                try:
                    result = await self.validate_url_entry(task['url_entry'])
                    return {
                        **task,
                        'status': 'completed',
                        'completed_at': datetime.now().isoformat(),
                        'result': result
                    }
                except Exception as e:
                    print(f"❌ Error processing task {task['task_id']}: {str(e)}")
                    return {
                        **task,
                        'status': 'failed',
                        'completed_at': datetime.now().isoformat(),
                        'result': {
                            **task['url_entry'],
                            'validated': True,
                            'is_valid': False,
                            'replacement_url': None,
                            'validation_notes': f"Validation failed: {str(e)}"
                        }
                    }

        # Process tasks concurrently with semaphore limiting
        tasks_coroutines = [process_task_with_semaphore(task) for task in tasks]
        processed_tasks = await asyncio.gather(*tasks_coroutines, return_exceptions=True)

        # Filter out exceptions and return successful results
        successful_tasks = [task for task in processed_tasks if not isinstance(task, Exception)]
        return successful_tasks
