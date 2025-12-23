#!/usr/bin/env python3
"""
Generic AI-Powered Content Validator Agent
Uses LLM intelligence with multiple tools to validate and find content.
"""

import json
import asyncio
import aiohttp
import aiofiles
import subprocess
import shlex
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import os
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests


class RateLimiter:
    """Simple rate limiter using asyncio primitives."""

    def __init__(self, rate_per_minute: float, burst_limit: int = 1):
        self.rate_per_minute = rate_per_minute
        self.burst_limit = burst_limit
        self.interval = 60.0 / rate_per_minute  # seconds between requests
        self.last_request_times = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()

            # Remove old timestamps outside the burst window
            cutoff_time = now - 60.0  # 1 minute window
            self.last_request_times = [t for t in self.last_request_times if t > cutoff_time]

            if len(self.last_request_times) >= self.burst_limit:
                # Calculate wait time based on the oldest request in our burst window
                wait_time = self.interval - (now - self.last_request_times[-self.burst_limit])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            self.last_request_times.append(time.time())

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


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
                "external_apis": {
                    "firecrawl": {"enabled": False},
                    "brave_search": {"enabled": False}
                }
            }


class GenericContentValidatorAgent:
    """Generic AI-powered agent that uses LLM intelligence with multiple tools."""

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

        # Read LLM settings from config
        self.max_iterations = self.config.get("lm_studio", {}).get("max_iterations", 5)

        # Setup detailed logging
        self.logger = logging.getLogger(f"URLValidatorAgent_{agent_id}")
        self.logger.setLevel(logging.DEBUG)

        # Create file handler for detailed logs
        log_filename = f"url_validation_agent_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Create console handler for important messages only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Agent {agent_id} initialized with log file: {log_filename}")

        # API configurations
        self.firecrawl_config = self.config.get('external_apis', {}).get('firecrawl', {})
        self.brave_config = self.config.get('external_apis', {}).get('brave_search', {})
        self.rate_limits = self.config.get('rate_limiting', {})

        # Rate limiters
        self.firecrawl_limiter = RateLimiter(
            rate_per_minute=self.rate_limits.get('firecrawl', {}).get('requests_per_minute', 10),
            burst_limit=self.rate_limits.get('firecrawl', {}).get('burst_limit', 3)
        )
        self.brave_limiter = RateLimiter(
            rate_per_minute=self.rate_limits.get('brave_search', {}).get('requests_per_minute', 5),
            burst_limit=self.rate_limits.get('brave_search', {}).get('burst_limit', 2)
        )

        self.max_tokens = self.config.get('lm_studio', {}).get('max_tokens', 2000)
        self.temperature = self.config.get('lm_studio', {}).get('temperature', 0.1)
        self.session: Optional[aiohttp.ClientSession] = None

        # Tool availability tracking
        self.api_available = {
            'firecrawl': self.firecrawl_config.get('enabled', False),
            'brave_search': self.brave_config.get('enabled', False),
            'curl': True,  # Always available
            'beautifulsoup': True  # Always available
        }

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

    def _get_available_tools(self) -> List[str]:
        """Get list of available tools."""
        tools = []
        if self.api_available['firecrawl']:
            tools.extend(['firecrawl_scrape', 'firecrawl_search', 'firecrawl_crawl'])
        if self.api_available['brave_search']:
            tools.extend(['brave_search'])
        tools.extend(['curl', 'beautifulsoup', 'regex_search'])
        return tools

    def _build_analysis_prompt(self, content_request: Dict[str, Any],
                             conversation_history: List[Dict],
                             available_tools: List[str],
                             iteration: int) -> str:
        """Build LLM prompt for content analysis."""

        task_description = content_request.get('task', 'validate_content')
        context = content_request.get('context', {})

        prompt = f"""You are an intelligent content validation agent with access to various tools.
Your task: {task_description}

Context:
{json.dumps(context, indent=2)}

Available tools: {', '.join(available_tools)}

Tool descriptions:
- curl: Make HTTP requests to check if URLs are accessible
- beautifulsoup: Parse HTML content and extract links/text
- regex_search: Search content with regular expressions
- firecrawl_scrape: Scrape web content from a single URL
- firecrawl_search: Search the web for information
- firecrawl_crawl: Crawl multiple pages on a website

"""

        if conversation_history:
            prompt += "\nPrevious attempts:\n"
            for i, hist in enumerate(conversation_history[-2:], 1):  # Show last 2 attempts
                prompt += f"Attempt {hist['iteration']}: {hist.get('reasoning', 'N/A')[:100]}\n"
                if 'tool_results' in hist and hist['tool_results']:
                    # Summarize key results instead of dumping everything
                    tool_summary = []
                    for tool_name, result in hist['tool_results'].items():
                        if result is None:
                            tool_summary.append(f"{tool_name}: No result")
                        elif isinstance(result, dict):
                            if 'error' in result and result['error'] is not None:
                                tool_summary.append(f"{tool_name}: ERROR - {str(result['error'])[:50]}")
                            elif 'status_code' in result:
                                tool_summary.append(f"{tool_name}: HTTP {result['status_code']}")
                            elif 'data' in result:
                                tool_summary.append(f"{tool_name}: Got data ({len(str(result['data']))} chars)")
                            else:
                                tool_summary.append(f"{tool_name}: Success")
                        else:
                            tool_summary.append(f"{tool_name}: {str(result)[:50]}")
                    prompt += f"Results: {'; '.join(tool_summary)}\n"
                if 'error' in hist:
                    prompt += f"Error: {hist['error'][:100]}\n"

        prompt += """

INSTRUCTIONS:
1. Use tools to gather information about the URL (check accessibility, content, etc.)
2. After gathering sufficient information, ALWAYS conclude with a final answer
3. Do NOT call tools repeatedly without making progress toward a conclusion
4. You MUST reach a conclusion by the 3rd tool call at latest

For URL validation tasks, intelligently assess content quality based on URL type:

**Exercise URLs (url_type: exercise)**:
- Must contain specific, actionable exercises or puzzles, not just listing/overview pages
- GOOD exercises contain actual problems, code to implement, or specific challenges
- BAD exercises are just navigation pages, lists, or general descriptions
- If firecrawl_scrape fails, use curl to get basic content and analyze it
- Look for specific exercise indicators vs general listing indicators

**Video URLs (url_type: video)**:
- Must link to specific videos, not channels, playlists, or video listings

**Article URLs (url_type: article)**:
- Can be specific articles or curated lists with educational value

**Tool Strategy**:
1. First try firecrawl_scrape for detailed content analysis
2. If firecrawl_scrape fails (API errors, etc.), immediately try curl for basic validation
3. Use content analysis to determine if exercise URLs are specific or just listings
4. Conclude based on content quality, not just accessibility

To use a tool, use this format:
<|channel|>commentary to=TOOL_NAME <|constrain|>json<|message|>{"param": "value"}
REASONING: Brief explanation of why you're using this tool

For your FINAL CONCLUSION, use the Harmony format:
<|channel|>final <|constrain|>json<|message|>{"conclusion": "[Brief summary: URL is valid or URL is invalid - [reason] or URL is invalid, replacement: [specific_exercise_url]]"}

**CRITICAL FALLBACK RULE**: If you see firecrawl_scrape fail with "API error" or "Unauthorized", you MUST try curl next.
**CRITICAL FOR EXERCISES**: Exercise URLs need content verification - if firecrawl fails, use curl and analyze the HTML for exercise-specific content.

After 1-2 tool calls, you should have enough information to make a decision. What would you like to do next?"""

        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with retry logic."""
        self.logger.debug(f"Calling LLM with prompt (length: {len(prompt)} chars)")
        self.logger.debug(f"LLM Prompt: {prompt[:500]}..." if len(prompt) > 500 else f"LLM Prompt: {prompt}")

        max_retries = 3

        for attempt in range(max_retries):
            try:
                self.logger.debug(f"LLM call attempt {attempt+1}/{max_retries}")
                # For Harmony format, we need system and developer messages
                system_content = """You are an intelligent content validation agent.
Reasoning: high
# Valid channels: commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'firecrawl_scrape', 'curl'."""

                developer_content = """# Instructions
You are a URL validation agent. Use tools to check if URLs are accessible and valid.

# Tools
## firecrawl_scrape
type firecrawl_scrape = (_: {url: string}) => any;

## curl
type curl = (_: {url: string}) => any;"""

                request_data = {
                    "model": "openai/gpt-oss-20b",
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "developer", "content": developer_content},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
                self.logger.debug(f"LLM request data: {json.dumps(request_data, indent=2)}")

                async with self.session.post(
                    f"{self.lm_studio_url}/v1/chat/completions",
                    json=request_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_content = result['choices'][0]['message']['content'].strip()
                        self.logger.debug(f"LLM response (length: {len(response_content)} chars): {response_content[:500]}..." if len(response_content) > 500 else f"LLM response: {response_content}")
                        return response_content
                    else:
                        error_msg = f"LLM API error (attempt {attempt+1}): {response.status}"
                        self.logger.warning(error_msg)
                        print(f"❌ {error_msg}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                error_msg = f"LLM call failed (attempt {attempt+1}): {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                print(f"❌ {error_msg}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        error_msg = "LLM call failed after all retries"
        self.logger.error(error_msg)
        raise Exception(error_msg)

    def _parse_llm_response(self, response: str) -> Tuple[str, str, List[Dict]]:
        """Parse LLM response for action and tool calls using Harmony format."""
        response = response.strip()

        # Check for Harmony final answer format: <|channel|>final <|constrain|>json<|message|>CONCLUSION_JSON
        import re
        final_answer_pattern = r'<\|channel\|>final <\|constrain\|>json<\|message\|>(.+)$'
        match = re.search(final_answer_pattern, response, re.DOTALL)
        if match:
            try:
                final_data = json.loads(match.group(1).strip())
                if 'conclusion' in final_data:
                    # Create a synthetic FINAL_ANSWER response for compatibility
                    synthetic_response = f"FINAL_ANSWER\nCONCLUSION: {final_data['conclusion']}"
                    return 'final_answer', synthetic_response, []
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON from Harmony final format: {match.group(1)}")

        if response.startswith('FINAL_ANSWER'):
            return 'final_answer', response, []

        # Check for the custom tool call format: <|channel|>commentary to=TOOL_NAME <|constrain|>json<|message|>PARAMS_JSON
        custom_tool_pattern = r'<\|channel\|>commentary to=(\w+) <\|constrain\|>json<\|message\|>(.+)$'
        match = re.search(custom_tool_pattern, response, re.DOTALL)

        if match:
            tool_name = match.group(1)
            try:
                params = json.loads(match.group(2).strip())
                tool_calls = [{
                    'tool': tool_name,
                    'params': params,
                    'reasoning': f'Using {tool_name} tool as requested by LLM'
                }]
                return 'use_tools', response, tool_calls
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON params from custom tool format: {match.group(2)}")

        # Look for tool usage in the expected format
        tool_calls = []
        lines = response.split('\n')

        current_tool = None
        current_params = None
        current_reasoning = None

        for line in lines:
            line = line.strip()
            if line.startswith('TOOL:'):
                if current_tool:
                    tool_calls.append({
                        'tool': current_tool,
                        'params': current_params or {},
                        'reasoning': current_reasoning or ''
                    })
                current_tool = line[5:].strip()
                current_params = None
                current_reasoning = None
            elif line.startswith('PARAMS:'):
                try:
                    current_params = json.loads(line[7:].strip())
                except:
                    current_params = {}
            elif line.startswith('REASONING:'):
                current_reasoning = line[10:].strip()

        # Add final tool if exists
        if current_tool:
            tool_calls.append({
                'tool': current_tool,
                'params': current_params or {},
                'reasoning': current_reasoning or ''
            })

        action = 'use_tools' if tool_calls else 'continue'
        return action, response, tool_calls

    async def _execute_tools(self, tool_calls: List[Dict]) -> Dict[str, Any]:
        """Execute the requested tools."""
        self.logger.info(f"Executing {len(tool_calls)} tool calls")
        results = {}

        for tool_call in tool_calls:
            tool_name = tool_call['tool']
            params = tool_call['params']
            reasoning = tool_call['reasoning']

            self.logger.info(f"Executing tool: {tool_name}")
            self.logger.debug(f"Tool reasoning: {reasoning}")
            self.logger.debug(f"Tool params: {json.dumps(params, indent=2)}")
            print(f"🔧 Executing tool: {tool_name} - {reasoning}")

            start_time = time.time()
            try:
                if tool_name == 'curl':
                    results[tool_name] = await self._tool_curl(params)
                elif tool_name == 'beautifulsoup':
                    results[tool_name] = await self._tool_beautifulsoup(params)
                elif tool_name == 'regex_search':
                    results[tool_name] = self._tool_regex_search(params)
                elif tool_name.startswith('firecrawl_'):
                    results[tool_name] = await self._tool_firecrawl(tool_name, params)
                elif tool_name == 'brave_search':
                    results[tool_name] = await self._tool_brave_search(params)
                else:
                    error_msg = f"Unknown tool: {tool_name}"
                    self.logger.error(error_msg)
                    results[tool_name] = {"error": error_msg}

                end_time = time.time()
                execution_time = end_time - start_time
                self.logger.info(f"Tool {tool_name} completed in {execution_time:.2f}s")
                self.logger.debug(f"Tool {tool_name} result: {json.dumps(results[tool_name], indent=2, default=str)[:1000]}...")

            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                error_msg = f"Tool {tool_name} failed after {execution_time:.2f}s: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                results[tool_name] = {"error": str(e)}

        self.logger.info(f"Completed executing {len(tool_calls)} tools")
        return results

    async def _tool_curl(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute curl command."""
        url = params.get('url', '')
        headers = params.get('headers', {})
        timeout = params.get('timeout', 10)

        self.logger.debug(f"curl tool: URL={url}, timeout={timeout}, headers={headers}")

        if not url:
            self.logger.warning("curl tool: No URL provided")
            return {"error": "No URL provided"}

        cmd = ['curl', '-s', '--max-time', str(timeout), '-L', '-w', '%{http_code}']  # -L for redirects, -w for status

        # Add headers
        for key, value in headers.items():
            cmd.extend(['-H', f'{key}: {value}'])

        cmd.append(url)
        self.logger.debug(f"curl command: {' '.join(cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            output = stdout.decode('utf-8', errors='ignore')
            # With -w %{http_code}, the last 3 characters are the status code
            if len(output) >= 3:
                content = output[:-3]
                status_code = int(output[-3:])
            else:
                content = output
                status_code = 0

            result = {
                "status_code": status_code,
                "content": content,
                "error": stderr.decode('utf-8', errors='ignore') if stderr else None
            }

            content_length = len(result.get('content', ''))
            self.logger.debug(f"curl result: status={result['status_code']}, content_length={content_length}")
            if result.get('error'):
                self.logger.debug(f"curl stderr: {result['error']}")

            return result
        except Exception as e:
            self.logger.error(f"curl tool failed: {str(e)}", exc_info=True)
            return {"error": str(e)}

    async def _tool_beautifulsoup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse HTML content with BeautifulSoup."""
        html_content = params.get('html', '')
        selectors = params.get('selectors', {})

        if not html_content:
            return {"error": "No HTML content provided"}

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            results = {}

            # Extract title
            if soup.title:
                results['title'] = soup.title.string.strip()

            # Apply selectors
            for selector_name, selector in selectors.items():
                if selector.startswith('#'):  # ID selector
                    elements = soup.find_all(id=selector[1:])
                elif selector.startswith('.'):  # Class selector
                    elements = soup.find_all(class_=selector[1:])
                else:  # Tag selector
                    elements = soup.find_all(selector)

                results[selector_name] = [
                    {'text': elem.get_text().strip(), 'attrs': dict(elem.attrs)}
                    for elem in elements
                ]

            # Extract all links
            results['links'] = [
                {'text': a.get_text().strip(), 'href': a.get('href')}
                for a in soup.find_all('a', href=True)
            ]

            return results

        except Exception as e:
            return {"error": str(e)}

    def _tool_regex_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search content with regex patterns."""
        content = params.get('content', '')
        patterns = params.get('patterns', {})

        if not content:
            return {"error": "No content provided"}

        results = {}
        for pattern_name, pattern in patterns.items():
            try:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                results[pattern_name] = matches
            except Exception as e:
                results[pattern_name] = {"error": str(e)}

        return results

    async def _tool_firecrawl(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Firecrawl API calls with rate limiting."""
        async with self.firecrawl_limiter:
            return await self._execute_firecrawl_api(tool_name, params)

    async def _execute_firecrawl_api(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actual Firecrawl API call."""
        api_key = self.firecrawl_config.get('api_key')
        base_url = self.firecrawl_config.get('base_url', 'https://api.firecrawl.dev')

        self.logger.debug(f"Firecrawl API call: tool={tool_name}, params keys={list(params.keys())}")

        if not api_key:
            self.logger.error("Firecrawl API key not configured")
            return {"error": "Firecrawl API key not configured"}

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        endpoint_map = {
            'firecrawl_scrape': '/v1/scrape',
            'firecrawl_search': '/v1/search',
            'firecrawl_crawl': '/v1/crawl'
        }

        endpoint = endpoint_map.get(tool_name)
        if not endpoint:
            error_msg = f"Unknown Firecrawl tool: {tool_name}"
            self.logger.error(error_msg)
            return {"error": error_msg}

        full_url = f"{base_url}{endpoint}"
        self.logger.debug(f"Firecrawl request: {full_url}")

        try:
            async with self.session.post(
                full_url,
                headers=headers,
                json=params
            ) as response:
                self.logger.debug(f"Firecrawl response status: {response.status}")
                if response.status == 200:
                    result = await response.json()
                    self.logger.debug(f"Firecrawl success: result keys={list(result.keys()) if isinstance(result, dict) else type(result)}")
                    return result
                else:
                    error_text = await response.text()
                    error_msg = f"Firecrawl API error {response.status}: {error_text}"
                    self.logger.error(error_msg)
                    return {"error": error_msg}

        except Exception as e:
            error_msg = f"Firecrawl API call failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"error": str(e)}

    async def _tool_brave_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Brave Search API with rate limiting."""
        async with self.brave_limiter:
            return await self._execute_brave_search_api(params)

    async def _execute_brave_search_api(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute actual Brave Search API call."""
        api_key = self.brave_config.get('api_key')
        base_url = self.brave_config.get('base_url', 'https://api.search.brave.com')

        if not api_key:
            return {"error": "Brave Search API key not configured"}

        query = params.get('query', '')
        if not query:
            return {"error": "No search query provided"}

        try:
            url = f"{base_url}/res/v1/web/search"
            params_dict = {
                'q': query,
                'count': params.get('count', 10),
                'country': params.get('country', 'US')
            }

            headers = {'X-Subscription-Token': api_key}

            async with self.session.get(url, params=params_dict, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    return {"error": f"Brave Search API error {response.status}: {error_text}"}

        except Exception as e:
            return {"error": str(e)}

    def _extract_final_answer(self, tool_results: Dict[str, Any],
                            conversation_history: List[Dict]) -> Dict[str, Any]:
        """Extract final answer from tool results and determine URL validity."""

        # Get URL type from the first conversation item (the original request)
        url_type = "unknown"
        if conversation_history and conversation_history[0].get('llm_prompt'):
            # Extract from the prompt which contains the context
            prompt = conversation_history[0]['llm_prompt']
            if '"url_type": "exercise"' in prompt:
                url_type = "exercise"

        # Analyze tool results to determine if URL is valid
        is_valid = False
        replacement_url = None
        reason = "Analysis completed"

        # Check firecrawl_scrape results first (has content analysis)
        if 'firecrawl_scrape' in tool_results:
            scrape_result = tool_results['firecrawl_scrape']
            if scrape_result and 'data' in scrape_result and scrape_result['data']:
                # If we got data from scraping, the URL is likely valid
                is_valid = True
                reason = "URL is accessible and content was successfully retrieved"
            elif scrape_result and 'error' in scrape_result:
                is_valid = False
                reason = f"Content analysis failed: {scrape_result['error']}"

        # Check curl results (basic accessibility only)
        elif 'curl' in tool_results:
            curl_result = tool_results['curl']
            if curl_result and 'status_code' in curl_result:
                status_code = curl_result['status_code']
                if 200 <= status_code < 400:  # Success or redirect
                    # For exercise URLs, we ideally need content analysis, but basic accessibility is a start
                    if url_type == "exercise":
                        is_valid = False  # Conservative approach: require content analysis for exercises
                        reason = f"URL is accessible (HTTP {status_code}) but needs content analysis to verify it's a specific exercise, not just a listing"
                        # Suggest trying firecrawl for proper content analysis
                        content = curl_result.get('content', '')
                        replacement_url = self._suggest_specific_exercise_url(content)
                    else:
                        # For non-exercise URLs, basic accessibility is sufficient
                        is_valid = True
                        reason = f"URL is accessible (HTTP {status_code})"
                else:
                    is_valid = False
                    reason = f"URL returned HTTP {status_code}"
            elif curl_result and 'error' in curl_result:
                is_valid = False
                reason = f"URL validation failed: {curl_result['error']}"

        # If no tools were successful, assume invalid
        if not tool_results:
            is_valid = False
            reason = "No validation data collected"

        return {
            'result': {
                'is_valid': is_valid,
                'replacement_url': replacement_url,
                'reason': reason,
                'tool_results': tool_results
            },
            'conversation': conversation_history,
            'status': 'completed'
        }

    def _is_exercise_listing_page(self, html_content: str) -> bool:
        """Check if HTML content appears to be a general exercise listing page."""
        if not html_content:
            return False

        content_lower = html_content.lower()

        # Look for listing page indicators
        listing_indicators = [
            'challenges', 'exercises', 'problems', 'puzzles',
            'phase 1', 'phase 2', 'phase 3', 'beginner', 'intermediate', 'advanced',
            'gpu architecture', 'parallel computing', 'cuda programming'
        ]

        # Look for specific exercise indicators (good signs)
        specific_indicators = [
            'vector addition', 'matrix multiplication', 'kernel fusion',
            'shared memory', 'atomic operations', 'warp divergence',
            'implement a kernel', 'write cuda code', 'solve this puzzle'
        ]

        listing_score = sum(1 for indicator in listing_indicators if indicator in content_lower)
        specific_score = sum(1 for indicator in specific_indicators if indicator in content_lower)

        # If it has more listing indicators than specific ones, it's likely a listing page
        return listing_score > specific_score and listing_score > 2

    def _suggest_specific_exercise_url(self, html_content: str) -> Optional[str]:
        """Try to suggest a more specific exercise URL based on content."""
        # This is a basic implementation - could be enhanced with better parsing
        content_lower = html_content.lower()

        # Look for LeetGPU specific patterns
        if 'leetgpu.com' in content_lower:
            # Common specific exercise URLs
            suggestions = [
                'https://leetgpu.com/challenges/vector-addition',
                'https://leetgpu.com/challenges/matrix-multiplication',
                'https://leetgpu.com/challenges/shared-memory'
            ]
            return suggestions[0]  # Return first suggestion

        # Look for GPU-Puzzles specific patterns
        if 'gpu-puzzles' in content_lower or 'srush' in content_lower:
            return 'https://github.com/srush/GPU-Puzzles/blob/main/GPU_puzzlers.ipynb'

        return None

    def _extract_direct_answer(self, llm_response: str,
                             conversation_history: List[Dict]) -> Dict[str, Any]:
        """Extract direct answer from LLM response."""
        return {
            'result': {'llm_answer': llm_response},
            'conversation': conversation_history,
            'status': 'completed'
        }

    def _create_fallback_result(self, content_request: Dict[str, Any],
                              conversation_history: List[Dict]) -> Dict[str, Any]:
        """Create fallback result when max iterations reached."""
        return {
            'result': {'fallback': True, 'reason': 'Max iterations reached'},
            'conversation': conversation_history,
            'status': 'incomplete'
        }

    def _has_successful_tool_results(self, tool_results: Dict[str, Any]) -> bool:
        """Check if tool results indicate successful data retrieval."""
        for tool_name, result in tool_results.items():
            if result is None:
                continue

            # Check firecrawl_scrape results
            if tool_name == 'firecrawl_scrape':
                if isinstance(result, dict) and 'success' in result and result['success']:
                    if 'data' in result and result['data']:
                        return True

            # Check curl results
            elif tool_name == 'curl':
                if isinstance(result, dict) and 'status_code' in result:
                    status_code = result['status_code']
                    if 200 <= status_code < 400:  # Success or redirect
                        return True

        return False

    async def validate_content_generic(self, content_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic content validation using LLM intelligence with multiple tools.

        Args:
            content_request: Dictionary containing validation requirements

        Returns:
            Dictionary with validation results
        """
        return await self.analyze_content_generic(content_request)

    async def analyze_content_generic(self, content_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic content analysis using LLM intelligence with multiple tools.

        Args:
            content_request: Dictionary containing content analysis requirements

        Returns:
            Dictionary with analysis results and any found content
        """
        self.logger.info("Starting generic content analysis")
        self.logger.debug(f"Content request: {json.dumps(content_request, indent=2)}")

        max_iterations = self.max_iterations
        iteration = 0

        conversation_history = []
        available_tools = self._get_available_tools()
        self.logger.debug(f"Available tools: {available_tools}")

        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"Starting iteration {iteration}/{max_iterations}")

            # Create LLM prompt with current state
            prompt = self._build_analysis_prompt(content_request, conversation_history, available_tools, iteration)
            self.logger.debug(f"Iteration {iteration} prompt created (length: {len(prompt)} chars)")

            try:
                # Get LLM response
                llm_response = await self._call_llm(prompt)

                # Parse LLM response for tool usage and reasoning
                action, reasoning, tool_calls = self._parse_llm_response(llm_response)

                self.logger.info(f"Iteration {iteration} - Action: {action}, Tool calls: {len(tool_calls)}")
                self.logger.debug(f"Iteration {iteration} - Reasoning: {reasoning}")
                self.logger.debug(f"Iteration {iteration} - Tool calls: {json.dumps(tool_calls, indent=2)}")

                conversation_history.append({
                    'iteration': iteration,
                    'llm_prompt': prompt,
                    'llm_response': llm_response,
                    'action': action,
                    'reasoning': reasoning,
                    'tool_calls': tool_calls
                })

                # Execute tools if requested
                if tool_calls:
                    tool_results = await self._execute_tools(tool_calls)
                    conversation_history[-1]['tool_results'] = tool_results

                    # Check if we have a final answer from LLM response
                    if action == 'final_answer':
                        self.logger.info("Final answer reached after tool execution")
                        return self._extract_final_answer(tool_results, conversation_history)

                    # If we have successful tool results, force conclusion (curl gives basic validation)
                    if self._has_successful_tool_results(tool_results):
                        self.logger.info("Forcing final conclusion after successful tool results")
                        return self._extract_final_answer(tool_results, conversation_history)

                    # Also check if we have successful results from previous iterations
                    all_tool_results = {}
                    for hist in conversation_history:
                        if 'tool_results' in hist and hist['tool_results']:
                            all_tool_results.update(hist['tool_results'])

                    if self._has_successful_tool_results(all_tool_results):
                        self.logger.info("Forcing final conclusion based on previous successful tool results")
                        return self._extract_final_answer(all_tool_results, conversation_history)

                else:
                    # LLM provided direct answer
                    if action == 'final_answer':
                        self.logger.info("Final answer provided directly by LLM")
                        return self._extract_direct_answer(llm_response, conversation_history)

            except Exception as e:
                error_msg = f"Iteration {iteration} failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                print(f"❌ {error_msg}")
                conversation_history.append({
                    'iteration': iteration,
                    'error': str(e)
                })

        # Return best effort result
        self.logger.warning(f"Max iterations ({max_iterations}) reached, returning fallback result")
        return self._create_fallback_result(content_request, conversation_history)

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

    async def process_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process validation tasks using the generic content validator.

        Args:
            tasks: List of task dictionaries

        Returns:
            List of processed tasks with results
        """
        self.logger.info(f"Starting to process {len(tasks)} tasks")
        processed_tasks = []

        for i, task in enumerate(tasks):
            task_id = task.get('task_id', f'task_{i}')
            url_entry = task.get('url_entry', {})
            url = url_entry.get('url', 'unknown')
            url_type = url_entry.get('url_type', 'unknown')
            topic = url_entry.get('topic_title', 'unknown')

            self.logger.info(f"[{task_id}] Processing task {i+1}/{len(tasks)}: {url} (type: {url_type})")
            print(f"🔍 Processing task: {url}")

            try:
                # Convert task to generic content request
                content_request = {
                    'task': 'validate_and_find_replacement',
                    'context': {
                        'original_url': url_entry.get('url'),
                        'url_type': url_entry.get('url_type'),
                        'topic': url_entry.get('topic_title'),
                        'phase': url_entry.get('phase_name'),
                        'group': url_entry.get('group_title'),
                        'is_broken': url_entry.get('is_broken', False)
                    }
                }

                self.logger.debug(f"[{task_id}] Content request: {json.dumps(content_request, indent=2)}")

                # Use generic analysis
                start_time = time.time()
                analysis_result = await self.analyze_content_generic(content_request)
                end_time = time.time()

                analysis_duration = end_time - start_time
                self.logger.info(f"[{task_id}] Analysis completed in {analysis_duration:.2f}s")

                # Convert result back to expected format
                result = {
                    'topic_id': url_entry.get('topic_id'),
                    'url_type': url_entry.get('url_type'),
                    'original_url': url_entry.get('url'),
                    'is_valid': analysis_result.get('result', {}).get('is_valid', False),
                    'replacement_url': analysis_result.get('result', {}).get('replacement_url'),
                    'reason': analysis_result.get('result', {}).get('reason', 'Analysis completed'),
                    'analysis_data': analysis_result,
                    'processing_duration': analysis_duration
                }

                self.logger.info(f"[{task_id}] Result: valid={result['is_valid']}, replacement={result['replacement_url'] is not None}")
                self.logger.debug(f"[{task_id}] Full result: {json.dumps(result, indent=2, default=str)}")

                processed_tasks.append({
                    'task': task,
                    'result': result
                })

            except Exception as e:
                self.logger.error(f"[{task_id}] Task processing failed: {str(e)}", exc_info=True)
                print(f"❌ Task processing failed: {str(e)}")
                processed_tasks.append({
                    'task': task,
                    'result': {
                        'topic_id': url_entry.get('topic_id'),
                        'url_type': url_entry.get('url_type'),
                        'original_url': url_entry.get('url'),
                        'is_valid': False,
                        'replacement_url': None,
                        'reason': f"Processing failed: {str(e)}",
                        'error': str(e)
                    }
                })

        self.logger.info(f"Completed processing {len(processed_tasks)} tasks")
        return processed_tasks
