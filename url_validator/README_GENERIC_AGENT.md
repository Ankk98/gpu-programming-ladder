# Generic AI-Powered Content Validator Agent

## Overview

The URL validation system has been completely refactored into a **Generic AI-Powered Content Validator Agent** that uses LLM intelligence with multiple tools to handle a wide range of content validation and discovery tasks.

## Key Changes

### 1. Removed MCP Tools
- Eliminated dependency on MCP (Model Context Protocol) tools
- Direct API integration instead of MCP servers

### 2. Removed Specific URL Handling
- No more hardcoded logic for LeetGPU, GitHub, or other platforms
- Generic approach that works with any website or content type

### 3. LLM-Driven Loop System
- LLM runs in iterative loops with complete freedom to use available tools
- Intelligent tool selection based on task requirements
- Conversational approach where LLM can request multiple tools in sequence

### 4. Comprehensive API Rate Limiting
- Built-in rate limiting for all APIs (Firecrawl, Brave Search)
- Exponential backoff on failures
- Configurable limits per API

### 5. Robust Fallback System
When APIs are unavailable or rate-limited, the system falls back to:
- **curl**: Direct HTTP requests with full control
- **regex**: Pattern matching on raw content
- **BeautifulSoup**: HTML parsing and DOM manipulation

## Available Tools

The agent has access to the following tools:

### API Tools (when enabled)
- `firecrawl_scrape`: Scrape specific URLs with JavaScript rendering
- `firecrawl_search`: Web search capabilities
- `firecrawl_crawl`: Crawl entire websites
- `brave_search`: Alternative search API

### Always Available Tools
- `curl`: Direct HTTP requests with custom headers, redirects, timeouts
- `beautifulsoup`: HTML parsing, selector extraction, link discovery
- `regex_search`: Pattern matching on any content

## Configuration

```json
{
  "external_apis": {
    "firecrawl": {
      "enabled": true,
      "api_key": "your-key",
      "base_url": "https://api.firecrawl.dev"
    },
    "brave_search": {
      "enabled": false,
      "api_key": "your-key",
      "base_url": "https://api.search.brave.com"
    }
  },
  "rate_limiting": {
    "firecrawl": {
      "requests_per_minute": 10,
      "burst_limit": 3,
      "backoff_seconds": 60
    },
    "brave_search": {
      "requests_per_minute": 5,
      "burst_limit": 2,
      "backoff_seconds": 30
    }
  }
}
```

## How It Works

### LLM Loop Process

1. **Task Analysis**: LLM analyzes the content request and determines what information is needed
2. **Tool Selection**: LLM chooses appropriate tools based on available options and task requirements
3. **Tool Execution**: Agent executes the requested tools with proper rate limiting
4. **Result Integration**: LLM processes tool results and decides next steps
5. **Iteration**: Process continues until LLM reaches a final conclusion or max iterations

### Example Interaction

```
User Task: "Validate and find replacement for broken GPU programming tutorial"

LLM Analysis: "Need to check if URL exists, then search for similar content"

Tool Request: "Use curl to check URL existence"
Result: URL returns 404

LLM Decision: "URL is broken, need to find replacement"

Tool Request: "Use brave_search to find GPU programming tutorials"
Result: List of alternative URLs

LLM Decision: "Found potential replacements, validate them"

Tool Request: "Use curl to check replacement URLs"
Result: Valid URLs found

Final Answer: "Here are working replacement URLs..."
```

## Usage Examples

### Basic Content Validation
```python
content_request = {
    'task': 'validate_content',
    'context': {
        'url': 'https://example.com/tutorial',
        'topic': 'GPU Programming',
        'expected_type': 'tutorial'
    }
}

result = await agent.validate_content_generic(content_request)
```

### Advanced Content Discovery
```python
content_request = {
    'task': 'find_similar_content',
    'context': {
        'original_topic': 'CUDA Memory Management',
        'broken_url': 'https://old-site.com/cuda-mem',
        'platforms': ['github.com', 'nvidia.com', 'medium.com']
    }
}

result = await agent.analyze_content_generic(content_request)
```

## Benefits

1. **Flexibility**: Handles any content type or validation task
2. **Resilience**: Multiple fallback options when APIs fail
3. **Intelligence**: LLM makes smart decisions about tool usage
4. **Scalability**: Built-in rate limiting prevents API abuse
5. **Extensibility**: Easy to add new tools or APIs
6. **Compatibility**: No dependency on potentially outdated third-party libraries

## Files Modified

- `url_validator_agent.py`: Complete rewrite with generic agent
- `config.json`: Removed MCP config, added rate limiting
- `requirements.txt`: Added rate limiting and API client libraries
- `url_validation_orchestrator.py`: Updated to use new agent
- `test_llm_analysis.py`: Updated for new interface

## Demo

Run `python demo_generic_agent.py` to see the system in action with:
- Tool availability check
- URL validation
- LLM-driven content analysis
- Individual tool testing

The system is now a powerful, generic AI agent that can handle virtually any web content validation or discovery task using intelligent tool selection and LLM reasoning.
