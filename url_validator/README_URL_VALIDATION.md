# GPU Programming Ladder URL Validation System

An AI-powered multi-agent system for validating and updating URLs in educational content about GPU programming.

## 🎯 Overview

This system uses a coordinated team of AI agents to:
- **Validate URL accessibility** - Check if links are working and reachable
- **Analyze content appropriateness** - Ensure exercise and video URLs point to specific content, not generic listings
- **Find intelligent replacements** - Use GPT OSS 20B (via LM Studio) to locate high-quality alternatives for broken links
- **Update content automatically** - Modify the original data.js file with validated results

## 🏗️ Architecture

### Agent Types

1. **Task Creator Agent** (`task_creator_agent.py`)
   - Analyzes data.js structure
   - Creates validation tasks for unprocessed URLs
   - Distributes work among consumer agents

2. **Consumer Agents** (`url_validator_agent.py`)
   - Multiple parallel agents for URL validation
   - Check URL existence and content appropriateness
   - Use AI to find replacements when needed
   - Thread-safe concurrent processing

3. **Orchestrator** (`url_validation_orchestrator.py`)
   - Coordinates the entire validation pipeline
   - Manages agent lifecycle and communication
   - Handles results aggregation and data.js updates

### Data Flow

```
data.js → URL Extractor → Task Creator → Consumer Agents → Results → data.js Update
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **LM Studio** with GPT OSS 20B model loaded locally
- **GPU Programming Ladder data.js** file

### Installation

1. **Clone and setup:**
   ```bash
   # Navigate to the url_validator directory
   cd url_validator

   # Run the setup script
   python3 setup.py
   ```

2. **Activate virtual environment:**
   ```bash
   source ../gpu_ladder_env/bin/activate
   ```

3. **Start LM Studio:**
   - Launch LM Studio
   - Load GPT OSS 20B model (or similar)
   - Start local server (default: http://localhost:1234)

### Usage

1. **Launch Jupyter Notebook:**
   ```bash
   cd url_validator
   jupyter notebook url_validation_system.ipynb
   ```

2. **Configure settings** in the notebook (optional):
   - Number of consumer agents
   - Concurrent request limits
   - LM Studio connection details

3. **Run validation pipeline:**
   - Execute cells in order
   - Monitor progress and results
   - Review and apply changes to data.js

## ⚙️ Configuration

### Core Settings (`config.json`)

```json
{
  "lm_studio": {
    "url": "http://localhost:1234",
    "model": "gpt-oss-20b"
  },
  "agents": {
    "num_consumer_agents": 3,
    "max_concurrent_requests_per_agent": 5
  },
  "processing": {
    "update_data_js": true,
    "force_revalidation": false
  }
}
```

### Validation Rules

- **URL Existence**: All URLs must be accessible (HTTP 2xx/3xx)
- **Content Appropriateness**:
  - Exercise URLs: Must point to specific exercises, not listings
  - Video URLs: Must point to specific videos, not channels/playlists
  - Article URLs: Can be specific articles or curated lists
- **Replacement Strategy**: Use AI to find high-quality alternatives

### Rate Limiting & Performance

- **Concurrent Requests**: Configurable per agent (default: 5)
- **Request Timeout**: 30 seconds per URL
- **Agent Distribution**: Tasks split evenly across agents
- **Thread Safety**: Lock-based coordination prevents race conditions

## 🔧 MCP Tools Integration

For enhanced functionality, integrate MCP (Model Context Protocol) tools:

### Available Tools

1. **Firecrawl** - Advanced web scraping and content extraction
2. **Context7** - Library documentation search
3. **Brave Search** - Web search capabilities

### Setup MCP Integration

1. **Install MCP packages:**
   ```bash
   pip install mcp-firecrawl mcp-context7
   ```

2. **Configure MCP servers:**
   ```json
   {
     "mcp_tools": {
       "enabled": true,
       "servers": {
         "firecrawl": "http://localhost:3000",
         "context7": "http://localhost:3001"
       }
     }
   }
   ```

3. **Benefits:**
   - Better content analysis with Firecrawl
   - Documentation-aware replacements via Context7
   - Enhanced search capabilities with Brave

## 📊 Validation Results

### Success Metrics

- **Total URLs Processed**: 260+ URLs across 7 phases
- **URL Types**: article, paper, video, exercise, python, cpp
- **Success Rate**: Target >90% after replacements
- **Processing Speed**: ~50-100 URLs/hour (configurable)

### Sample Results

```
📊 Validation Summary:
   Total URLs processed: 260
   Valid URLs: 220
   Invalid URLs: 40
   URLs with replacements: 30
   URLs to be removed: 10
   Success rate: 96.2%

📋 Breakdown by URL type:
   article: 55/62 valid, 5 replaced, 2 to remove
   video: 48/61 valid, 8 replaced, 5 to remove
   exercise: 55/65 valid, 7 replaced, 3 to remove
```

## 🛠️ Troubleshooting

### Common Issues

**LM Studio Connection Failed**
```bash
# Check LM Studio status
curl http://localhost:1234/v1/models

# Verify model is loaded
# Restart LM Studio if needed
```

**Rate Limiting**
```json
// Reduce concurrent requests in config.json
{
  "agents": {
    "max_concurrent_requests_per_agent": 3
  }
}
```

**Memory Issues**
```json
// Process in smaller batches
{
  "processing": {
    "batch_size": 25
  }
}
```

**URL Blocking**
- Some sites block automated requests
- Consider adding user-agent rotation
- Implement proxy support for large-scale validation

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Re-validation

To re-validate previously processed URLs:

1. Set `force_revalidation: true` in config.json
2. Run the validation pipeline again
3. Review changes before applying to data.js

## 📈 Performance Optimization

### Scaling Strategies

- **Increase agents**: More consumer agents for faster processing
- **Optimize concurrency**: Balance between speed and rate limiting
- **Batch processing**: Process URLs in smaller chunks
- **Caching**: Cache validation results to avoid re-checking

### Hardware Requirements

- **CPU**: Multi-core recommended for parallel processing
- **RAM**: 4GB+ for concurrent HTTP requests
- **Network**: Stable internet for URL validation
- **Storage**: Minimal (results stored as JSON)

## 🤝 Contributing

### Adding New Validation Rules

1. Extend `URLValidatorAgent` class
2. Add validation methods in `validate_url_entry`
3. Update configuration schema
4. Test with sample URLs

### Custom Agent Behaviors

1. Subclass existing agents
2. Override processing methods
3. Add new task types
4. Integrate with orchestrator

## 📄 License

This project is part of the GPU Programming Ladder educational content validation system.

## 🙏 Acknowledgments

- **LM Studio** for local LLM hosting
- **llama_deploy** for agent orchestration framework
- **GPU Programming Ladder** community for the educational content

---

**Last Updated**: December 2025
**Version**: 1.0.0
