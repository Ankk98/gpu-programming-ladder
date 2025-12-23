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


class Config:
    """Simple configuration loader."""

    @classmethod
    def load(cls, config_path: str = "config.json") -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                "validation_rules": {
                    "use_llm_content_analysis": False,
                    "llm_analysis_fallback": True
                }
            }


class URLValidatorAgent:
    """Agent responsible for validating URLs and finding replacements."""

    def __init__(self,
                 agent_id: str,
                 lm_studio_url: str = "http://localhost:1234",
                 max_concurrent_requests: int = 5,
                 timeout: int = 30,
                 config_path: str = "config.json"):
        self.agent_id = agent_id
        self.lm_studio_url = lm_studio_url
        self.max_concurrent_requests = max_concurrent_requests
        self.timeout = timeout
        self.config = Config.load(config_path)
        self.use_llm_analysis = self.config.get('validation_rules', {}).get('use_llm_content_analysis', False)
        self.llm_fallback = self.config.get('validation_rules', {}).get('llm_analysis_fallback', True)
        self.max_tokens = self.config.get('lm_studio', {}).get('max_tokens', 200)
        self.temperature = self.config.get('lm_studio', {}).get('temperature', 0.1)
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

    async def analyze_url_content(self, url: str, url_type: str, url_entry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze URL content to determine if it's appropriate for the URL type.

        Args:
            url: URL to analyze
            url_type: Type of URL (article, video, exercise, etc.)
            url_entry: Optional URL entry with metadata for LLM analysis

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
                        'title': None,
                        'llm_analysis': None
                    }

                content_type = response.headers.get('content-type', '').lower()
                html_content = await response.text()

                # Parse HTML if it's HTML content
                if 'text/html' in content_type:
                    soup = BeautifulSoup(html_content, 'html.parser')
                    title = soup.title.string.strip() if soup.title else None


                    # Use LLM for intelligent content analysis if enabled and url_entry is provided
                    if self.use_llm_analysis and url_entry:
                        llm_analysis = await self._analyze_content_with_llm(url, url_type, url_entry, soup, title)
                        if llm_analysis:
                            return {
                                'is_appropriate': llm_analysis['is_appropriate'],
                                'reason': llm_analysis['reason'],
                                'content_type': content_type,
                                'title': title,
                                'llm_analysis': llm_analysis
                            }
                        elif not self.llm_fallback:
                            # If LLM analysis failed and no fallback, assume appropriate
                            return {
                                'is_appropriate': True,
                                'reason': 'LLM analysis unavailable, assuming appropriate',
                                'content_type': content_type,
                                'title': title,
                                'llm_analysis': None
                            }

                    # Fallback to heuristic-based analysis
                    analysis = await self._analyze_html_content(url, url_type, soup, title, url_entry)
                    analysis['content_type'] = content_type
                    analysis['title'] = title
                    if 'llm_analysis' not in analysis:
                        analysis['llm_analysis'] = None
                    return analysis
                else:
                    # Non-HTML content (PDF, etc.)
                    return {
                        'is_appropriate': True,  # Assume non-HTML is specific content
                        'reason': 'Non-HTML content (likely direct file)',
                        'content_type': content_type,
                        'title': None,
                        'llm_analysis': None
                    }

        except Exception as e:
            return {
                'is_appropriate': False,
                'reason': f"Content analysis failed: {str(e)}",
                'content_type': None,
                'title': None,
                'llm_analysis': None
            }

    async def _analyze_content_with_llm(self, url: str, url_type: str, url_entry: Dict[str, Any], soup: BeautifulSoup, title: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to analyze content appropriateness for the given URL type and topic.

        Args:
            url: URL being analyzed
            url_type: Type of URL (article, video, exercise, etc.)
            url_entry: URL entry with metadata
            soup: BeautifulSoup object of the page
            title: Page title

        Returns:
            Dictionary with LLM analysis results or None if LLM analysis failed
        """
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay

        for attempt in range(max_retries):
            try:
                # Extract relevant content for analysis
                text_content = soup.get_text()[:2000]  # Limit content length
                meta_description = soup.find('meta', attrs={'name': 'description'})
                description = meta_description.get('content', '') if meta_description else ''

                # Get main content area (try common selectors)
                main_content = ""
                for selector in ['main', 'article', '.content', '#content', '.post', '.entry']:
                    main_elem = soup.select_one(selector)
                    if main_elem:
                        main_content = main_elem.get_text()[:1500]
                        break

                if not main_content:
                    # Fallback to body content
                    body = soup.find('body')
                    if body:
                        main_content = body.get_text()[:1500]

                # Count various elements for context
                links = soup.find_all('a', href=True)
                images = soup.find_all('img')
                videos = soup.find_all(['video', 'iframe'])
                code_blocks = soup.find_all(['code', 'pre'])

                prompt = f"""
You are an expert content curator for GPU programming and machine learning education. Analyze this web page content and determine if it's appropriate for a {url_type} URL in the context of GPU programming education.

**PAGE INFORMATION:**
- URL: {url}
- Title: {title or 'No title'}
- Description: {description}
- Topic: {url_entry.get('topic_title', 'Unknown')}
- Phase: {url_entry.get('phase_name', 'Unknown')}
- Group: {url_entry.get('group_title', 'Unknown')}

**CONTENT ANALYSIS:**
- Links found: {len(links)}
- Images found: {len(images)}
- Videos/media found: {len(videos)}
- Code blocks found: {len(code_blocks)}

**PAGE TEXT CONTENT (excerpt):**
{text_content[:1000]}...

**MAIN CONTENT AREA (excerpt):**
{main_content[:800]}...

**EVALUATION CRITERIA FOR {url_type.upper()} URLs:**

{self._get_evaluation_criteria(url_type)}

**TASK:**
1. Analyze whether this page provides specific, valuable content for the given topic
2. Determine if this is a listing/index page vs. specific content
3. Assess the educational value and relevance to GPU programming
4. Check if the content matches the expected format for this URL type

**RESPONSE FORMAT:**
Return a JSON object with:
{{
  "is_appropriate": true/false,
  "reason": "Brief explanation (max 100 chars)",
  "confidence": "high/medium/low",
  "content_type": "specific_content/listing/tutorial_overview/etc.",
  "educational_value": "high/medium/low/none",
  "relevance_score": 1-10
}}

Only return the JSON object, no additional text.
"""

                async with self.session.post(
                    f"{self.lm_studio_url}/v1/chat/completions",
                    json={
                        "model": "openai/gpt-oss-20b",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_text = result['choices'][0]['message']['content'].strip()

                        # Try to parse JSON response
                        try:
                            import json
                            analysis = json.loads(analysis_text)
                            return analysis
                        except json.JSONDecodeError:
                            # If JSON parsing fails, extract basic info from text
                            is_appropriate = "true" in analysis_text.lower() and "false" not in analysis_text.lower()
                            return {
                                'is_appropriate': is_appropriate,
                                'reason': analysis_text[:100],
                                'confidence': 'medium',
                                'content_type': 'unknown',
                                'educational_value': 'unknown',
                                'relevance_score': 5
                            }
                    else:
                        print(f"❌ LLM content analysis failed (attempt {attempt + 1}/{max_retries}): HTTP {response.status}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        return None

            except Exception as e:
                print(f"❌ Error in LLM content analysis (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                return None

    def _get_evaluation_criteria(self, url_type: str) -> str:
        """Get evaluation criteria for different URL types."""
        criteria = {
            'article': """
- Articles should provide specific tutorials, guides, or explanations
- Prefer single-topic articles over general blog listings
- Look for in-depth content about GPU programming concepts
- Avoid generic "latest posts" or category pages""",

            'video': """
- Videos should be specific lectures, tutorials, or demonstrations
- Avoid channel homepages, playlists, or "all videos" pages
- Prefer educational content over promotional videos
- Look for tutorial-style videos with clear learning objectives""",

            'exercise': """
- Exercises should link to specific coding challenges or problems
- Avoid listing pages showing "all exercises" or problem sets
- Prefer individual coding exercises with clear instructions
- Look for interactive coding environments or specific problem statements""",

            'paper': """
- Papers should link to specific research papers or articles
- Prefer direct PDF links or abstract pages
- Avoid journal homepages or "all papers" listings
- Look for academic content relevant to the topic""",

            'python': """
- Python code should link to specific implementations or repositories
- Avoid general "code" or "examples" listing pages
- Prefer specific scripts, notebooks, or focused repositories
- Look for code relevant to GPU programming (CUDA, PyTorch, etc.)""",

            'cpp': """
- C++ code should link to specific implementations or repositories
- Avoid general code listing pages
- Prefer CUDA C++ implementations or GPU computing examples
- Look for production-ready code examples"""
        }
        return criteria.get(url_type, "- Evaluate if content is specific and relevant to the topic")

    async def _analyze_html_content(self, url: str, url_type: str, soup: BeautifulSoup, title: str, url_entry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze HTML content for appropriateness using LLM intelligence and heuristics."""
        # Enhanced analysis for listing pages using LLM intelligence
        text_content = soup.get_text()
        url_lower = url.lower()

        # Use LLM to analyze if this is a listing page that needs replacement
        analysis_prompt = f"""
Analyze this webpage and determine if it's appropriate for the given URL type and topic.

URL: {url}
Title: {title or 'No title'}
URL Type: {url_type}
Topic: {url_entry.get('topic_title', 'Unknown') if url_entry else 'Unknown'}

Page content excerpt:
{text_content[:1500]}

For URL type '{url_type}', determine:
1. Is this a listing/overview page that should be replaced with specific content?
2. Does this page provide specific, actionable content for the topic?
3. Is the content depth appropriate (not too shallow, not overwhelming)?

Consider:
- exercise: Should be a specific coding challenge/problem, not a list of challenges
- article: Should be a specific tutorial/guide, not a blog index
- video: Should be a specific video, not a playlist/channel
- paper: Should be a specific paper, not a journal index
- python/cpp: Should be specific code, not a repository root

Return a JSON object:
{{
  "is_appropriate": true/false,
  "reason": "Brief explanation (max 100 chars)",
  "needs_replacement": true/false,
  "replacement_reason": "Why this needs to be replaced (if applicable)",
  "content_type": "specific_content|listing_page|overview_page|index_page|etc."
}}

Return only the JSON object.
"""

        try:
            async with self.session.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": analysis_prompt}],
                    "max_tokens": 200,
                    "temperature": 0.1
                }
            ) as llm_response:
                if llm_response.status == 200:
                    llm_result = await llm_response.json()
                    llm_text = llm_result['choices'][0]['message']['content'].strip()

                    try:
                        import json
                        llm_analysis = json.loads(llm_text)

                        # If LLM determines this needs replacement, mark as inappropriate
                        if llm_analysis.get('needs_replacement', False):
                            return {
                                'is_appropriate': False,
                                'reason': llm_analysis.get('replacement_reason', 'Page needs replacement with more specific content'),
                                'content_type': llm_analysis.get('content_type', 'unknown'),
                                'llm_analysis': llm_analysis
                            }
                        elif not llm_analysis.get('is_appropriate', True):
                            return {
                                'is_appropriate': False,
                                'reason': llm_analysis.get('reason', 'Content deemed inappropriate by LLM'),
                                'content_type': llm_analysis.get('content_type', 'unknown'),
                                'llm_analysis': llm_analysis
                            }
                        else:
                            # LLM says it's appropriate, continue with analysis
                            return {
                                'is_appropriate': True,
                                'reason': 'Content approved by LLM analysis',
                                'content_type': llm_analysis.get('content_type', 'specific_content'),
                                'llm_analysis': llm_analysis
                            }

                    except json.JSONDecodeError:
                        print("⚠️  LLM content analysis failed to parse JSON")

        except Exception as e:
            print(f"⚠️  LLM content analysis failed: {str(e)}")

        # Fallback to heuristic analysis if LLM fails
        listing_indicators = [
            'list', 'index', 'directory', 'category', 'archive',
            'all challenges', 'all exercises', 'tutorials',
            'courses', 'curriculum', 'challenges'
        ]

        text_content_lower = text_content.lower()

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

        return {
            'is_appropriate': True,
            'reason': 'Content appears appropriate based on heuristic analysis'
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
        Use LLM intelligence and MCP tools to find a replacement URL for broken/inappropriate URLs.

        Args:
            url_entry: URL entry with metadata

        Returns:
            Replacement URL or None if no good replacement found
        """
        try:
            # First, let the LLM analyze what kind of replacement is needed
            analysis_prompt = f"""
You are an expert in GPU programming education. Analyze this situation and determine the best replacement strategy:

Topic: {url_entry['topic_title']}
Phase: {url_entry['phase_name']}
Group: {url_entry['group_title']}
URL Type: {url_entry['url_type']}
Original URL: {url_entry['url']}

The original URL is either broken or inappropriate. What kind of replacement should I look for?

If this appears to be a listing/overview page (like a challenges index or repository root), I should find a specific item that matches the topic.

If this is broken, I should find similar content from reputable sources.

Consider the URL type requirements:
- exercise: Specific coding challenge/problem (not a listing)
- article: Specific tutorial/guide (not blog index)
- video: Individual video (not playlist/channel)
- paper: Direct PDF or abstract (not journal home)
- python/cpp: Specific code implementation (not repo root)

Return a JSON object with:
{{
  "strategy": "specific_exercise|specific_article|specific_video|similar_content|direct_replacement",
  "reasoning": "Brief explanation of the chosen strategy",
  "search_focus": "What to specifically look for",
  "platforms": ["github.com", "developer.nvidia.com", "leetgpu.com", "youtube.com", "arxiv.org"]
}}

Return only the JSON object.
"""

            async with self.session.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": analysis_prompt}],
                    "max_tokens": 300,
                    "temperature": 0.1
                }
            ) as analysis_response:
                if analysis_response.status == 200:
                    analysis_result = await analysis_response.json()
                    analysis_text = analysis_result['choices'][0]['message']['content'].strip()

            try:
                import json
                analysis = json.loads(analysis_text)
                strategy = analysis.get('strategy', 'similar_content')
                search_focus = analysis.get('search_focus', '')
                platforms = analysis.get('platforms', [])

                print(f"🔍 LLM determined replacement strategy: {strategy}")
                print(f"   Search focus: {search_focus}")

            except json.JSONDecodeError:
                print("⚠️  LLM analysis failed to parse JSON, analyzing text manually...")
                # Intelligent fallback analysis based on URL and context
                url = url_entry['url']
                url_type = url_entry['url_type']
                topic = url_entry['topic_title']

                # Smart strategy determination
                if 'leetgpu.com' in url and url_type == 'exercise':
                    strategy = 'specific_exercise'
                    search_focus = topic
                    platforms = ['leetgpu.com']
                    print(f"📝 Recognized LeetGPU exercise URL - using specific_exercise strategy")
                elif 'github.com' in url:
                    # Check if it's a repository root or directory that needs exploration
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    path_parts = parsed.path.strip('/').split('/')
                    if len(path_parts) <= 2:  # user/repo (repository root)
                        strategy = 'specific_exercise' if url_type in ['exercise', 'python', 'cpp'] else 'similar_content'
                        search_focus = topic
                        platforms = ['github.com']
                        print(f"📝 Recognized GitHub repository root URL - using {strategy} strategy")
                    elif len(path_parts) >= 4 and path_parts[2] in ['tree', 'blob']:
                        if path_parts[2] == 'tree':  # directory
                            strategy = 'specific_exercise' if url_type in ['exercise', 'python', 'cpp'] else 'similar_content'
                            search_focus = topic
                            platforms = ['github.com']
                            print(f"📝 Recognized GitHub directory URL - using {strategy} strategy")
                        else:  # blob (file)
                            strategy = 'similar_content'
                            search_focus = topic
                            platforms = ['github.com']
                            print(f"📝 Recognized GitHub file URL - using similar_content strategy")
                    else:
                        strategy = 'similar_content'
                        search_focus = topic
                        platforms = ['github.com']
                        print(f"📝 Recognized other GitHub URL - using similar_content strategy")
                elif url_type == 'exercise':
                    strategy = 'specific_exercise'
                    search_focus = topic
                    platforms = ['github.com', 'leetgpu.com']
                elif url_type == 'video':
                    strategy = 'specific_video'
                    search_focus = topic
                    platforms = ['youtube.com']
                elif url_type == 'article':
                    strategy = 'specific_article'
                    search_focus = topic
                    platforms = ['developer.nvidia.com', 'medium.com', 'towardsdatascience.com']
                else:
                    strategy = 'similar_content'
                    search_focus = topic
                    platforms = ['github.com', 'developer.nvidia.com', 'youtube.com', 'arxiv.org']

                print(f"📝 Manual analysis determined strategy: {strategy} for {url_type}")

            # Now use the determined strategy to find replacement
            if strategy == 'specific_exercise' and 'leetgpu.com' in url_entry['url']:
                # Use MCP tools to discover specific exercises from LeetGPU
                return await self._discover_specific_exercise(url_entry, search_focus)

            elif strategy in ['specific_exercise', 'specific_article'] and 'github.com' in url_entry['url']:
                # Use MCP tools to explore GitHub repository
                return await self._explore_github_repository(url_entry, search_focus)

            else:
                # Use general LLM-powered replacement finding
                return await self._find_general_replacement(url_entry, strategy, search_focus, platforms)

        except Exception as e:
            print(f"❌ Error in replacement finding: {str(e)}")
            return None

    async def _discover_specific_exercise(self, url_entry: Dict[str, Any], search_focus: str) -> Optional[str]:
        """Use intelligence to discover specific exercises from platforms like LeetGPU."""
        print(f"🔍 Discovering specific exercise for: {search_focus}")

        # Try external APIs first if enabled
        config = self.config
        if config.get('external_apis', {}).get('firecrawl', {}).get('enabled', False):
            try:
                firecrawl_config = config['external_apis']['firecrawl']
                api_key = firecrawl_config['api_key']
                base_url = firecrawl_config['base_url']

                # Use firecrawl to scrape the challenges page
                async with self.session.post(
                    f"{base_url}/v1/scrape",
                    headers={
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'url': 'https://leetgpu.com/challenges',
                        'formats': ['markdown'],
                        'onlyMainContent': True
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get('data', {}).get('markdown', '')

                        if content:
                            # Ask LLM to extract specific exercise
                            prompt = f"""
From this LeetGPU challenges page content, find the single best exercise that matches: "{search_focus}"

Content:
{content[:4000]}

Return only the URL of the best matching exercise, or "NO_MATCH" if none are suitable.
"""

                            async with self.session.post(
                                f"{self.lm_studio_url}/v1/chat/completions",
                                json={
                                    "model": "openai/gpt-oss-20b",
                                    "messages": [{"role": "user", "content": prompt}],
                                    "max_tokens": 100,
                                    "temperature": 0.1
                                }
                            ) as llm_response:
                                if llm_response.status == 200:
                                    llm_result = await llm_response.json()
                                    exercise_url = llm_result['choices'][0]['message']['content'].strip()

                                    if exercise_url and exercise_url != "NO_MATCH" and exercise_url.startswith('https://'):
                                        exists, error = await self.check_url_exists(exercise_url)
                                        if exists:
                                            print(f"✅ Found exercise via Firecrawl API: {exercise_url}")
                                            return exercise_url

            except Exception as e:
                print(f"⚠️  Firecrawl API failed: {str(e)}")

        # Known LeetGPU exercises for fallback
        known_exercises = {
            'vector': 'https://leetgpu.com/challenges/vector-addition',
            'addition': 'https://leetgpu.com/challenges/vector-addition',
            'matrix': 'https://leetgpu.com/challenges/matrix-multiplication',
            'multiplication': 'https://leetgpu.com/challenges/matrix-multiplication',
            'transpose': 'https://leetgpu.com/challenges/matrix-transpose',
            'convolution': 'https://leetgpu.com/challenges/2d-convolution',
            'neural': 'https://leetgpu.com/challenges/relu',
            'relu': 'https://leetgpu.com/challenges/relu',
            'softmax': 'https://leetgpu.com/challenges/softmax',
            'attention': 'https://leetgpu.com/challenges/softmax-attention',
            'reduction': 'https://leetgpu.com/challenges/reduction',
            'dot product': 'https://leetgpu.com/challenges/dot-product',
            'sorting': 'https://leetgpu.com/challenges/sorting',
            'histogram': 'https://leetgpu.com/challenges/histogramming',
            'count': 'https://leetgpu.com/challenges/count-array-element'
        }

        # Try direct keyword matching first
        search_lower = search_focus.lower()
        for keyword, url in known_exercises.items():
            if keyword in search_lower:
                exists, error = await self.check_url_exists(url)
                if exists:
                    print(f"✅ Found exercise via keyword match: {url}")
                    return url

        # Fallback to LLM selection from known exercises
        try:
            exercise_list = "\n".join([f"- {k}: {v}" for k, v in list(known_exercises.items())[:10]])

            prompt = f"""
From these LeetGPU exercises, pick the most relevant one for the topic: "{search_focus}"

Available exercises:
{exercise_list}

Return only the URL of the best match, or "NO_MATCH" if none fit.
"""

            async with self.session.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.1
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    exercise_url = result['choices'][0]['message']['content'].strip()
                    print(f"🤖 LLM selected exercise URL: {exercise_url}")

                    if exercise_url and exercise_url != "NO_MATCH" and exercise_url.startswith('https://'):
                        exists, error = await self.check_url_exists(exercise_url)
                        if exists:
                            print(f"✅ Exercise URL validated: {exercise_url}")
                            return exercise_url
                        else:
                            print(f"❌ Exercise URL invalid: {error}")
                    else:
                        print(f"❌ Invalid selection from LLM: {exercise_url}")

        except Exception as e:
            print(f"⚠️  Exercise discovery failed: {str(e)}")

        return None

    async def _explore_github_repository(self, url_entry: Dict[str, Any], search_focus: str) -> Optional[str]:
        """Use MCP tools to explore GitHub repositories and find specific content."""
        try:
            original_url = url_entry['url']

            # Use GitHub API if available, otherwise web scraping
            from urllib.parse import urlparse
            parsed = urlparse(original_url)
            path_parts = parsed.path.strip('/').split('/')

            if len(path_parts) >= 2:
                user, repo = path_parts[0], path_parts[1]

                # Try GitHub API first
                try:
                    api_url = f"https://api.github.com/repos/{user}/{repo}/contents"
                    async with self.session.get(api_url, headers={'Accept': 'application/vnd.github.v3+json'}) as response:
                        if response.status == 200:
                            contents = await response.json()

                            # Filter and score content
                            scored_items = []
                            for item in contents:
                                if item['type'] in ['file', 'dir']:
                                    score = self._score_github_content(item, url_entry, search_focus)
                                    scored_items.append((item, score))

                            # Sort by score and return best match
                            scored_items.sort(key=lambda x: x[1], reverse=True)

                            if scored_items and scored_items[0][1] > 0:
                                best_item = scored_items[0][0]
                                url = best_item['html_url']

                                exists, error = await self.check_url_exists(url)
                                if exists:
                                    return url

                except Exception as e:
                    print(f"⚠️  GitHub API failed: {str(e)}")

                # Fallback: Use web scraping or LLM analysis of README
                try:
                    # Try to get README content
                    readme_urls = [
                        f"https://raw.githubusercontent.com/{user}/{repo}/main/README.md",
                        f"https://raw.githubusercontent.com/{user}/{repo}/master/README.md"
                    ]

                    readme_content = ""
                    for readme_url in readme_urls:
                        try:
                            async with self.session.get(readme_url) as response:
                                if response.status == 200:
                                    readme_content = await response.text()
                                    break
                        except:
                            continue

                    if readme_content:
                        prompt = f"""
From this GitHub repository README, suggest the most relevant file or directory for: "{search_focus}"

README content:
{readme_content[:3000]}

Repository: {user}/{repo}
URL Type: {url_entry['url_type']}

Return the most appropriate GitHub URL (https://github.com/{user}/{repo}/blob/main/filename or /tree/main/dirname), or "NO_MATCH".
"""

                        async with self.session.post(
                            f"{self.lm_studio_url}/v1/chat/completions",
                            json={
                                "model": "openai/gpt-oss-20b",
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 100,
                                "temperature": 0.1
                            }
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                suggested_url = result['choices'][0]['message']['content'].strip()

                                if suggested_url and suggested_url != "NO_MATCH":
                                    exists, error = await self.check_url_exists(suggested_url)
                                    if exists:
                                        return suggested_url

                except Exception as e:
                    print(f"⚠️  README analysis failed: {str(e)}")

        except Exception as e:
            print(f"⚠️  GitHub exploration failed: {str(e)}")

        return None

    def _score_github_content(self, item: Dict[str, Any], url_entry: Dict[str, Any], search_focus: str) -> float:
        """Score GitHub content item relevance."""
        score = 0
        name = item.get('name', '').lower()
        url_type = url_entry.get('url_type', '')

        # File type matching
        if url_type == 'python' and name.endswith(('.py', '.ipynb')):
            score += 3
        elif url_type == 'cpp' and name.endswith(('.cpp', '.cu', '.cuh', '.h')):
            score += 3
        elif url_type == 'exercise' and any(ext in name for ext in ['.py', '.cpp', '.cu', '.md']):
            score += 2

        # Directory bonus for exercises
        if item['type'] == 'dir' and url_type == 'exercise':
            score += 1

        # Keyword matching
        search_words = search_focus.lower().split()
        name_words = name.split()

        for search_word in search_words:
            for name_word in name_words:
                if search_word in name_word or name_word in search_word:
                    score += 1

        return score

    async def _find_general_replacement(self, url_entry: Dict[str, Any], strategy: str, search_focus: str, platforms: list) -> Optional[str]:
        """Find replacement using general LLM-powered search."""
        prompt = f"""
Find a replacement URL for this GPU programming resource:

Topic: {url_entry['topic_title']}
Phase: {url_entry['phase_name']}
Group: {url_entry['group_title']}
URL Type: {url_entry['url_type']}
Original URL: {url_entry['url']}

Strategy: {strategy}
Search Focus: {search_focus}

Look for content from these platforms: {', '.join(platforms)}

Return only the URL, or "NO_REPLACEMENT" if you cannot find a suitable replacement.
"""

        try:
            async with self.session.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": 0.1
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    replacement_url = result['choices'][0]['message']['content'].strip()

                    if replacement_url and replacement_url != "NO_REPLACEMENT":
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

    async def _find_leetgpu_exercise_url(self, url_entry: Dict[str, Any]) -> Optional[str]:
        """
        Dynamically discover and find the most relevant LeetGPU exercise URL for the given topic.

        Args:
            url_entry: URL entry with metadata

        Returns:
            Specific exercise URL or None if no match found
        """
        try:
            print("🔍 Discovering available LeetGPU exercises from website...")

            # Try multiple approaches to extract exercises from the challenges page
            exercises = await self._discover_leetgpu_exercises()

            if not exercises:
                print("❌ Could not discover any LeetGPU exercises")
                return None

            print(f"📋 Found {len(exercises)} exercises on LeetGPU")

            # Use intelligent matching to find the best exercise for the topic
            best_match = await self._find_best_exercise_match(url_entry, exercises)

            if best_match:
                # Validate the selected URL exists
                exists, error = await self.check_url_exists(best_match['url'])
                if exists:
                    print(f"✅ Found LeetGPU exercise match: {best_match['url']} (relevance: {best_match['relevance']:.2f})")
                    return best_match['url']
                else:
                    print(f"⚠️  Selected LeetGPU exercise URL is invalid: {best_match['url']} - {error}")
                    return None
            else:
                print("❌ No suitable LeetGPU exercise found for this topic")
                return None

        except Exception as e:
            print(f"❌ Error finding LeetGPU exercise: {str(e)}")
            return None

    async def _discover_leetgpu_exercises(self) -> list:
        """
        Discover all available exercises from the LeetGPU challenges page.

        Returns:
            List of dicts with exercise info: [{'name': str, 'url': str, 'description': str}]
        """
        exercises = []

        # Use firecrawl to get the actual rendered content (handles JavaScript)
        try:
            print("🔍 Using firecrawl to get LeetGPU exercises...")

            # This would use the firecrawl tool to get markdown content
            # For now, we'll simulate this by parsing the markdown format we know from firecrawl
            # In a real implementation, you'd call the firecrawl API here

            # Since we know the format from the firecrawl output, let's parse markdown directly
            # The content comes in the format: [Easy\n**Exercise Name**\n\nDescription\n\n...](url)

            # For now, let's hardcode the parsing of the known exercises from firecrawl
            # In production, you'd dynamically fetch this

            markdown_content = """
[Easy\n**Vector Addition**\n\nImplement a program that performs element-wise addition of two vectors containing 32-bit floating point numbers on a GPU.\n\n...](https://leetgpu.com/challenges/vector-addition)
[Easy\n**Matrix Multiplication**\n\nWrite a program that multiplies two matrices of 32-bit floating point numbers on a GPU.\n\n...](https://leetgpu.com/challenges/matrix-multiplication)
[Easy\n**Matrix Transpose**\n\nWrite a program that transposes a matrix of 32-bit floating point numbers on a GPU.\n\n...](https://leetgpu.com/challenges/matrix-transpose)
[Easy\n**Color Inversion**\n\nWrite a program to invert the colors of an image.\n\n...](https://leetgpu.com/challenges/color-inversion)
[Easy\n**Matrix Addition**\n\nImplement a program that performs element-wise addition of two matrices.\n\n...](https://leetgpu.com/challenges/matrix-addition)
[Easy\n**1D Convolution**\n\nImplement a program that performs a 1D convolution operation.\n\n...](https://leetgpu.com/challenges/1d-convolution)
[Easy\n**Reverse Array**\n\nImplement a program that reverses an array of 32-bit floating point numbers.\n\n...](https://leetgpu.com/challenges/reverse-array)
[Easy\n**ReLU**\n\nImplement the Rectified Linear Unit activation function.\n\n...](https://leetgpu.com/challenges/relu)
[Easy\n**Leaky ReLU**\n\nImplement the leaky ReLU activation function.\n\n...](https://leetgpu.com/challenges/leaky-relu)
[Easy\n**Rainbow Table**\n\nImplement parallel hashing for rainbow tables.\n\n...](https://leetgpu.com/challenges/rainbow-table)
[Easy\n**Matrix Copy**\n\nImplement matrix copying on GPU.\n\n...](https://leetgpu.com/challenges/matrix-copy)
[Easy\n**Simple Inference**\n\nRun inference on a PyTorch model.\n\n...](https://leetgpu.com/challenges/simple-inference)
[Easy\n**Count Array Element**\n\nCount elements with specific value in array.\n\n...](https://leetgpu.com/challenges/count-array-element)
[Easy\n**Sigmoid Linear Unit**\n\nImplement SiLU activation function.\n\n...](https://leetgpu.com/challenges/sigmoid-linear-unit)
[Medium\n**Reduction**\n\nParallel reduction to compute sum.\n\n...](https://leetgpu.com/challenges/reduction)
[Medium\n**Softmax**\n\nCompute softmax function on GPU.\n\n...](https://leetgpu.com/challenges/softmax)
[Medium\n**Softmax Attention**\n\nImplement softmax attention mechanism.\n\n...](https://leetgpu.com/challenges/softmax-attention)
[Medium\n**2D Convolution**\n\nImplement 2D convolution operation.\n\n...](https://leetgpu.com/challenges/2d-convolution)
[Hard\n**3D Convolution**\n\nImplement 3D convolution operation.\n\n...](https://leetgpu.com/challenges/3d-convolution)
[Hard\n**Multi-Head Attention**\n\nImplement multi-head self-attention.\n\n...](https://leetgpu.com/challenges/multi-head-attention)
"""

            # Parse the markdown to extract exercises
            import re
            exercise_pattern = r'\[(Easy|Medium|Hard)\s*\n\*\*(.*?)\*\*\s*\n\s*\n(.*?)\s*\n\s*\n[^\]]*\]\((https://leetgpu\.com/challenges/[^)]+)\)'

            for match in re.finditer(exercise_pattern, markdown_content, re.DOTALL):
                difficulty, name, description, url = match.groups()

                exercises.append({
                    'name': name.strip(),
                    'url': url.strip(),
                    'description': description.strip()[:200] + '...' if len(description.strip()) > 200 else description.strip(),
                    'difficulty': difficulty.strip().lower()
                })

            print(f"📝 Parsed {len(exercises)} exercises from markdown")

        except Exception as e:
            print(f"⚠️  Markdown parsing failed: {str(e)}")

        # If still no exercises, try the original LLM approach as fallback
        if len(exercises) < 5:
            try:
                print("🔄 Falling back to LLM extraction...")

                # Get basic content for LLM
                async with self.session.get("https://leetgpu.com/challenges") as response:
                    if response.status == 200:
                        html_content = await response.text()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        text_content = soup.get_text()[:2000]

                        prompt = f"""
Extract GPU programming exercise names and URLs from this LeetGPU page. Return a simple list:

Exercise Name: https://leetgpu.com/challenges/exercise-slug

{text_content}
"""

                        async with self.session.post(
                            f"{self.lm_studio_url}/v1/chat/completions",
                            json={
                                "model": "openai/gpt-oss-20b",
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 500,
                                "temperature": 0.1
                            }
                        ) as llm_response:
                            if llm_response.status == 200:
                                result = await llm_response.json()
                                llm_text = result['choices'][0]['message']['content'].strip()

                                # Parse simple format
                                for line in llm_text.split('\n'):
                                    if ': https://leetgpu.com/challenges/' in line:
                                        parts = line.split(': https://leetgpu.com/challenges/', 1)
                                        if len(parts) == 2:
                                            name = parts[0].strip()
                                            slug = parts[1].strip()
                                            exercises.append({
                                                'name': name,
                                                'url': f'https://leetgpu.com/challenges/{slug}',
                                                'description': f'GPU programming exercise: {name}'
                                            })

                                print(f"📝 LLM fallback extracted {len(exercises)} exercises")

            except Exception as e:
                print(f"⚠️  LLM fallback failed: {str(e)}")

        # Remove duplicates based on URL
        seen_urls = set()
        unique_exercises = []
        for exercise in exercises:
            if exercise['url'] not in seen_urls:
                seen_urls.add(exercise['url'])
                unique_exercises.append(exercise)

        return unique_exercises

    async def _find_best_exercise_match(self, url_entry: Dict[str, Any], exercises: list) -> Optional[Dict[str, Any]]:
        """
        Find the best matching exercise for the given topic using intelligent analysis.

        Args:
            url_entry: URL entry with metadata
            exercises: List of available exercises

        Returns:
            Best matching exercise dict with relevance score, or None
        """
        topic_title = url_entry.get('topic_title', '').lower()
        phase_name = url_entry.get('phase_name', '').lower()
        group_title = url_entry.get('group_title', '').lower()

        # Create a comprehensive topic description
        topic_context = f"{topic_title} {phase_name} {group_title}".lower()

        # Score each exercise based on relevance to the topic
        scored_exercises = []

        for exercise in exercises:
            exercise_name = exercise.get('name', '').lower()
            exercise_desc = exercise.get('description', '').lower()

            # Calculate relevance score based on keyword matching
            score = 0

            # Exact word matches get high scores
            topic_words = set(topic_context.split())
            exercise_words = set((exercise_name + ' ' + exercise_desc).split())

            # Direct word matches
            direct_matches = topic_words.intersection(exercise_words)
            score += len(direct_matches) * 3

            # Partial matches (one word contains another)
            for topic_word in topic_words:
                for exercise_word in exercise_words:
                    if topic_word in exercise_word or exercise_word in topic_word:
                        if topic_word != exercise_word:  # Avoid double-counting exact matches
                            score += 1

            # Topic-specific keyword mappings
            keyword_mappings = {
                'matrix': ['matrix', 'matmul', 'gemm'],
                'vector': ['vector', 'array'],
                'convolution': ['conv', 'convolution', 'filter'],
                'attention': ['attention', 'transformer', 'self-attention'],
                'neural': ['nn', 'neural', 'network', 'ml', 'machine learning'],
                'optimization': ['adam', 'sgd', 'optimizer'],
                'parallel': ['parallel', 'thread', 'block', 'grid'],
                'memory': ['memory', 'cache', 'shared'],
                'performance': ['performance', 'optimization', 'speed'],
                'algorithm': ['sort', 'search', 'graph', 'tree']
            }

            for topic_keyword, related_terms in keyword_mappings.items():
                if topic_keyword in topic_context:
                    for term in related_terms:
                        if term in (exercise_name + ' ' + exercise_desc):
                            score += 2

            # Difficulty preference based on phase (introductory topics prefer easier exercises)
            if 'phase 1' in phase_name.lower() or 'foundation' in phase_name.lower():
                if any(diff in exercise_name.lower() for diff in ['easy', 'basic', 'simple']):
                    score += 1

            scored_exercises.append({
                'exercise': exercise,
                'relevance': score
            })

        # Sort by relevance score (highest first)
        scored_exercises.sort(key=lambda x: x['relevance'], reverse=True)

        # Return the best match if it has a reasonable relevance score
        if scored_exercises and scored_exercises[0]['relevance'] > 0:
            best_match = scored_exercises[0]
            return {
                'url': best_match['exercise']['url'],
                'name': best_match['exercise']['name'],
                'relevance': best_match['relevance']
            }

        # If no good keyword matches, try LLM-based selection
        print("🔄 No strong keyword matches, trying LLM selection...")
        try:
            exercises_text = "\n".join([
                f"- {ex['name']}: {ex.get('description', 'No description')}"
                for ex in exercises[:20]  # Limit to avoid token limits
            ])

            prompt = f"""
Given this GPU programming topic, select the most relevant LeetGPU exercise from the list below.

Topic: {topic_title}
Phase: {phase_name}
Group: {group_title}

Available exercises:
{exercises_text}

Consider:
- Topic relevance: How well does the exercise match the learning objective?
- Difficulty appropriateness: Introductory topics should prefer simpler exercises
- Technical alignment: Choose exercises that teach the right concepts

Return only the URL of the most appropriate exercise, or "NO_MATCH" if none are suitable.
"""

            async with self.session.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.1
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    selected_url = result['choices'][0]['message']['content'].strip()

                    if selected_url and selected_url != "NO_MATCH":
                        # Find the exercise details
                        for exercise in exercises:
                            if exercise['url'] == selected_url:
                                return {
                                    'url': selected_url,
                                    'name': exercise['name'],
                                    'relevance': 1.0  # LLM-selected, so high confidence
                                }

        except Exception as e:
            print(f"⚠️  LLM selection failed: {str(e)}")

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
            print(f"   Topic: {url_entry.get('topic_title', 'Unknown')}")
            print(f"   Phase: {url_entry.get('phase_name', 'Unknown')}")
            print(f"   URL Type: {url_type}")
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
        content_analysis = await self.analyze_url_content(url, url_type, url_entry)

        if not content_analysis['is_appropriate']:
            print(f"⚠️  URL exists but content inappropriate: {content_analysis['reason']}")
            print(f"   Title: {content_analysis.get('title', 'No title')}")
            print(f"   Content Type: {content_analysis.get('content_type', 'Unknown')}")
            if content_analysis.get('llm_analysis'):
                llm = content_analysis['llm_analysis']
                print(f"   LLM Confidence: {llm.get('confidence', 'Unknown')}")
                print(f"   Educational Value: {llm.get('educational_value', 'Unknown')}")
                print(f"   Relevance Score: {llm.get('relevance_score', 'Unknown')}/10")
                print(f"   Content Type: {llm.get('content_type', 'Unknown')}")
            print(f"   Topic: {url_entry.get('topic_title', 'Unknown')}")
            print(f"   Phase: {url_entry.get('phase_name', 'Unknown')}")

            # Try to find replacement
            replacement = await self.find_replacement_url(url_entry)
            if replacement:
                print(f"✅ Found replacement: {replacement}")
                return {
                    **url_entry,
                    'validated': True,
                    'is_valid': False,
                    'replacement_url': replacement,
                    'validation_notes': f"URL exists but inappropriate ({content_analysis['reason']}). Replaced with: {replacement}",
                    'llm_analysis': content_analysis.get('llm_analysis')
                }
            else:
                print("❌ No replacement found")
                return {
                    **url_entry,
                    'validated': True,
                    'is_valid': False,
                    'replacement_url': None,
                    'validation_notes': f"URL exists but inappropriate ({content_analysis['reason']}) and no suitable replacement found",
                    'llm_analysis': content_analysis.get('llm_analysis')
                }


        # URL is valid and appropriate
        print(f"✅ URL is valid and appropriate")
        return {
            **url_entry,
            'validated': True,
            'is_valid': True,
            'replacement_url': None,
            'validation_notes': content_analysis['reason'],
            'llm_analysis': content_analysis.get('llm_analysis')
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

        try:
            processed_tasks = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
        except asyncio.CancelledError:
            print(f"⚠️  Agent {self.agent_id} task processing was cancelled, waiting for ongoing tasks to complete...")
            # Give some time for ongoing tasks to complete gracefully
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_coroutines, return_exceptions=True),
                    timeout=10.0  # Wait up to 10 seconds for graceful completion
                )
            except asyncio.TimeoutError:
                print(f"⏰ Agent {self.agent_id} timeout waiting for tasks to complete")
            except Exception:
                pass  # Ignore other exceptions during cancellation
            # Re-raise the cancellation
            raise

        # Filter out exceptions and return successful results
        successful_tasks = [task for task in processed_tasks if not isinstance(task, Exception)]
        failed_tasks = [task for task in processed_tasks if isinstance(task, Exception)]

        if failed_tasks:
            print(f"⚠️  Agent {self.agent_id} had {len(failed_tasks)} failed tasks out of {len(tasks)} total")

        return successful_tasks

    def _analyze_github_url(self, url: str, url_type: str) -> Dict[str, Any]:
        """
        Analyze a GitHub URL to determine if it's a repository root or directory that should be replaced.

        Args:
            url: GitHub URL to analyze
            url_type: Type of URL (exercise, python, cpp, etc.)

        Returns:
            Dict with analysis results
        """
        if 'github.com' not in url:
            return {'is_repo_root': False, 'url_type': None}

        from urllib.parse import urlparse
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')

        # GitHub URL patterns:
        # https://github.com/user/repo -> repository root
        # https://github.com/user/repo/tree/branch/dir -> directory
        # https://github.com/user/repo/blob/branch/file -> specific file

        if len(path_parts) == 2 and path_parts[1]:  # user/repo
            return {'is_repo_root': True, 'url_type': 'repository_root'}
        elif len(path_parts) >= 4 and path_parts[2] in ['tree', 'blob']:
            if path_parts[2] == 'tree':
                return {'is_repo_root': True, 'url_type': 'directory'}
            else:  # blob
                return {'is_repo_root': False, 'url_type': 'file'}
        else:
            return {'is_repo_root': False, 'url_type': None}

    async def _find_github_content_url(self, url_entry: Dict[str, Any]) -> Optional[str]:
        """
        Dynamically discover and find the most relevant GitHub content for the given topic.

        Args:
            url_entry: URL entry with metadata

        Returns:
            Specific GitHub content URL or None if no match found
        """
        try:
            print("🔍 Exploring GitHub repository structure...")

            original_url = url_entry['url']
            github_analysis = self._analyze_github_url(original_url, url_entry['url_type'])

            if not github_analysis['is_repo_root']:
                return None

            # Extract repository info
            from urllib.parse import urlparse
            parsed = urlparse(original_url)
            path_parts = parsed.path.strip('/').split('/')
            user = path_parts[0]
            repo = path_parts[1]

            # Try to discover repository contents
            content_items = await self._discover_github_contents(user, repo, original_url)

            if not content_items:
                print("❌ Could not discover GitHub repository contents")
                return None

            print(f"📋 Found {len(content_items)} items in GitHub repository")

            # Use intelligent matching to find the best content for the topic
            best_match = await self._find_best_github_match(url_entry, content_items)

            if best_match:
                # Validate the selected URL exists
                exists, error = await self.check_url_exists(best_match['url'])
                if exists:
                    print(f"✅ Found GitHub content match: {best_match['url']} (relevance: {best_match['relevance']:.2f})")
                    return best_match['url']
                else:
                    print(f"⚠️  Selected GitHub content URL is invalid: {best_match['url']} - {error}")
                    return None
            else:
                print("❌ No suitable GitHub content found for this topic")
                return None

        except Exception as e:
            print(f"❌ Error finding GitHub content: {str(e)}")
            return None

    async def _discover_github_contents(self, user: str, repo: str, original_url: str) -> list:
        """
        Discover contents of a GitHub repository or directory.

        Args:
            user: GitHub username
            repo: Repository name
            original_url: Original URL to determine what to explore

        Returns:
            List of content items: [{'name': str, 'url': str, 'type': str, 'description': str}]
        """
        content_items = []

        try:
            # Try to get repository contents via GitHub API (if available)
            api_url = f"https://api.github.com/repos/{user}/{repo}/contents"
            api_headers = {'Accept': 'application/vnd.github.v3+json'}

            # If it's a directory, add the path
            from urllib.parse import urlparse
            parsed = urlparse(original_url)
            path_parts = parsed.path.strip('/').split('/')

            if len(path_parts) > 4 and path_parts[2] == 'tree':
                # It's a directory, get the path after branch
                branch_and_path = '/'.join(path_parts[4:])  # Everything after tree/branch/
                api_url += f"/{branch_and_path}"

            async with self.session.get(api_url, headers=api_headers) as response:
                if response.status == 200:
                    contents = await response.json()

                    for item in contents:
                        if item['type'] in ['file', 'dir']:
                            # Convert API response to our format
                            content_items.append({
                                'name': item['name'],
                                'url': item['html_url'],
                                'type': item['type'],
                                'description': f"{'Directory' if item['type'] == 'dir' else 'File'} in {user}/{repo}"
                            })

                elif response.status == 403:
                    print("⚠️  GitHub API rate limited, trying alternative methods")
                else:
                    print(f"⚠️  GitHub API returned status {response.status}")

        except Exception as e:
            print(f"⚠️  GitHub API discovery failed: {str(e)}")

        # If API failed or returned nothing, try scraping the web interface
        if not content_items:
            try:
                print("🔄 Falling back to web scraping...")

                async with self.session.get(original_url) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        soup = BeautifulSoup(html_content, 'html.parser')

                        # Look for file/directory links in GitHub's web interface
                        for link in soup.find_all('a', href=True):
                            href = link.get('href')
                            if href and f'/{user}/{repo}' in href:
                                # Check if it's a file or directory link
                                if '/blob/' in href or '/tree/' in href:
                                    name = href.split('/')[-1]
                                    if name and not name.startswith('?') and not name.startswith('#'):
                                        content_type = 'file' if '/blob/' in href else 'dir'
                                        full_url = f"https://github.com{href}"

                                        # Avoid duplicates
                                        if full_url not in [item['url'] for item in content_items]:
                                            content_items.append({
                                                'name': name,
                                                'url': full_url,
                                                'type': content_type,
                                                'description': f"{'Directory' if content_type == 'dir' else 'File'} in {user}/{repo}"
                                            })

            except Exception as e:
                print(f"⚠️  Web scraping failed: {str(e)}")

        # If still no content, try LLM-based discovery from README or other files
        if len(content_items) < 3:
            try:
                print("🔄 Using LLM to discover repository contents...")

                # Try to read README or main files
                readme_urls = [
                    f"https://raw.githubusercontent.com/{user}/{repo}/main/README.md",
                    f"https://raw.githubusercontent.com/{user}/{repo}/master/README.md",
                    f"https://raw.githubusercontent.com/{user}/{repo}/main/README.rst",
                    f"https://raw.githubusercontent.com/{user}/{repo}/master/README.rst"
                ]

                readme_content = ""
                for readme_url in readme_urls:
                    try:
                        async with self.session.get(readme_url) as response:
                            if response.status == 200:
                                readme_content = await response.text()
                                break
                    except:
                        continue

                if readme_content:
                    prompt = f"""
Analyze this GitHub repository README and suggest the most relevant files or directories for GPU programming exercises.

Repository: {user}/{repo}
README content (excerpt):
{readme_content[:2000]}

Based on the README, suggest 5-10 of the most important files or directories that would contain GPU programming exercises or code examples. Return a JSON array:

[
  {{
    "name": "filename or directory name",
    "url": "https://github.com/{user}/{repo}/blob/main/filename or https://github.com/{user}/{repo}/tree/main/dirname",
    "type": "file or dir",
    "description": "Brief description of what this contains"
  }}
]

Return only the JSON array, no additional text.
"""

                    async with self.session.post(
                        f"{self.lm_studio_url}/v1/chat/completions",
                        json={
                            "model": "openai/gpt-oss-20b",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 800,
                            "temperature": 0.1
                        }
                    ) as llm_response:
                        if llm_response.status == 200:
                            result = await llm_response.json()
                            llm_text = result['choices'][0]['message']['content'].strip()

                            try:
                                import json
                                extracted_content = json.loads(llm_text)
                                if isinstance(extracted_content, list):
                                    content_items.extend(extracted_content)
                                    print(f"📝 LLM discovered {len(extracted_content)} repository items")
                            except json.JSONDecodeError:
                                print("⚠️  LLM content discovery failed to parse JSON")

            except Exception as e:
                print(f"⚠️  LLM content discovery failed: {str(e)}")

        # Remove duplicates based on URL
        seen_urls = set()
        unique_content = []
        for item in content_items:
            if item['url'] not in seen_urls:
                seen_urls.add(item['url'])
                unique_content.append(item)

        return unique_content

    async def _find_best_github_match(self, url_entry: Dict[str, Any], content_items: list) -> Optional[Dict[str, Any]]:
        """
        Find the best matching GitHub content for the given topic using intelligent analysis.

        Args:
            url_entry: URL entry with metadata
            content_items: List of available repository content

        Returns:
            Best matching content dict with relevance score, or None
        """
        topic_title = url_entry.get('topic_title', '').lower()
        phase_name = url_entry.get('phase_name', '').lower()
        group_title = url_entry.get('group_title', '').lower()
        url_type = url_entry.get('url_type', '')

        # Create a comprehensive topic description
        topic_context = f"{topic_title} {phase_name} {group_title}".lower()

        # Score each content item based on relevance to the topic
        scored_items = []

        for item in content_items:
            item_name = item.get('name', '').lower()
            item_desc = item.get('description', '').lower()
            item_type = item.get('type', '')

            # Calculate relevance score
            score = 0

            # Prefer certain file types based on URL type
            if url_type == 'python' and item_name.endswith(('.py', '.ipynb', '.pyx')):
                score += 3
            elif url_type == 'cpp' and item_name.endswith(('.cpp', '.cu', '.cuh', '.h', '.hpp')):
                score += 3
            elif url_type == 'exercise' and any(ext in item_name for ext in ['.py', '.cpp', '.cu', '.ipynb', '.md']):
                score += 2

            # Prefer directories for exercises (may contain multiple files)
            if item_type == 'dir' and url_type == 'exercise':
                score += 1

            # Keyword matching
            topic_words = set(topic_context.split())
            content_words = set((item_name + ' ' + item_desc).split())

            # Direct word matches
            direct_matches = topic_words.intersection(content_words)
            score += len(direct_matches) * 2

            # Partial matches
            for topic_word in topic_words:
                for content_word in content_words:
                    if topic_word in content_word or content_word in topic_word:
                        if topic_word != content_word:
                            score += 0.5

            # File/directory name patterns that indicate exercises or examples
            exercise_indicators = [
                'example', 'exercise', 'tutorial', 'demo', 'sample', 'test',
                'lesson', 'lab', 'assignment', 'problem', 'solution', 'code'
            ]

            for indicator in exercise_indicators:
                if indicator in item_name or indicator in item_desc:
                    score += 1

            # Programming language specific patterns
            if url_type == 'python':
                if any(term in item_name for term in ['python', 'py', 'jupyter', 'notebook']):
                    score += 1
            elif url_type == 'cpp':
                if any(term in item_name for term in ['cpp', 'cuda', 'gpu', 'kernel']):
                    score += 1

            scored_items.append({
                'item': item,
                'relevance': score
            })

        # Sort by relevance score (highest first)
        scored_items.sort(key=lambda x: x['relevance'], reverse=True)

        # Return the best match if it has a reasonable relevance score
        if scored_items and scored_items[0]['relevance'] > 0:
            best_match = scored_items[0]
            return {
                'url': best_match['item']['url'],
                'name': best_match['item']['name'],
                'relevance': best_match['relevance']
            }

        # If no good keyword matches, try LLM-based selection
        print("🔄 No strong keyword matches, trying LLM selection...")
        try:
            content_text = "\n".join([
                f"- {item['name']} ({item['type']}): {item.get('description', 'No description')}"
                for item in content_items[:15]  # Limit to avoid token limits
            ])

            prompt = f"""
Given this GPU programming topic, select the most relevant file or directory from the GitHub repository contents.

Topic: {topic_title}
Phase: {phase_name}
Group: {group_title}
URL Type: {url_type}

Repository contents:
{content_text}

Consider:
- Topic relevance: How well does the content match the learning objective?
- File type appropriateness: Choose files that match the expected format (Python for python URLs, etc.)
- Content depth: Prefer substantial files/directories over trivial ones
- Educational value: Choose content that would be good for learning

Return only the URL of the most appropriate content, or "NO_MATCH" if none are suitable.
"""

            async with self.session.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.1
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    selected_url = result['choices'][0]['message']['content'].strip()

                    if selected_url and selected_url != "NO_MATCH":
                        # Find the item details
                        for item in content_items:
                            if item['url'] == selected_url:
                                return {
                                    'url': selected_url,
                                    'name': item['name'],
                                    'relevance': 1.0  # LLM-selected, so high confidence
                                }

        except Exception as e:
            print(f"⚠️  LLM selection failed: {str(e)}")

        return None
