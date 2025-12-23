#!/usr/bin/env python3
"""
Test script for LLM-based content analysis functionality.
"""

import asyncio
import json
from url_validator_agent import URLValidatorAgent


async def test_llm_content_analysis():
    """Test the LLM content analysis feature."""
    print("🧪 Testing LLM Content Analysis")

    # Sample URL entry
    test_url_entry = {
        'topic_title': 'CUDA Memory Management',
        'phase_name': 'Phase 1: Foundations',
        'group_title': 'GPU Architecture',
        'url_type': 'article',
        'url': 'https://developer.nvidia.com/blog/cuda-pro-tip-always-set-current-device-explicitly/',
        'topic_id': 'test_topic_1'
    }

    # Test with LLM analysis enabled
    print("\n1️⃣ Testing with LLM analysis enabled...")

    async with URLValidatorAgent(
        agent_id="test_agent",
        lm_studio_url="http://localhost:1234",
        config_path="config.json"
    ) as agent:
        print(f"   LLM Analysis Enabled: {agent.use_llm_analysis}")
        print(f"   Fallback Enabled: {agent.llm_fallback}")

        # Test URL existence first
        exists, error = await agent.check_url_exists(test_url_entry['url'])
        if exists:
            print("   ✅ URL exists, testing content analysis...")

            # Test content analysis
            analysis = await agent.analyze_url_content(
                test_url_entry['url'],
                test_url_entry['url_type'],
                test_url_entry
            )

            print(f"   Content Analysis Result: {analysis['is_appropriate']}")
            print(f"   Reason: {analysis['reason']}")
            print(f"   LLM Analysis: {analysis.get('llm_analysis') is not None}")

            if analysis.get('llm_analysis'):
                llm_data = analysis['llm_analysis']
                print(f"   Confidence: {llm_data.get('confidence', 'N/A')}")
                print(f"   Educational Value: {llm_data.get('educational_value', 'N/A')}")
                print(f"   Relevance Score: {llm_data.get('relevance_score', 'N/A')}")
        else:
            print(f"   ❌ URL does not exist: {error}")

    # Test with LLM analysis disabled
    print("\n2️⃣ Testing with LLM analysis disabled...")

    # Temporarily modify config for testing
    with open('config.json', 'r') as f:
        config = json.load(f)

    original_llm_setting = config['validation_rules']['use_llm_content_analysis']
    config['validation_rules']['use_llm_content_analysis'] = False

    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

    try:
        async with URLValidatorAgent(
            agent_id="test_agent_no_llm",
            lm_studio_url="http://localhost:1234",
            config_path="config.json"
        ) as agent:
            print(f"   LLM Analysis Enabled: {agent.use_llm_analysis}")

            if exists:  # If URL existed from previous test
                analysis = await agent.analyze_url_content(
                    test_url_entry['url'],
                    test_url_entry['url_type'],
                    test_url_entry
                )

                print(f"   Content Analysis Result: {analysis['is_appropriate']}")
                print(f"   Reason: {analysis['reason']}")
                print(f"   LLM Analysis: {analysis.get('llm_analysis') is not None}")

    finally:
        # Restore original config
        config['validation_rules']['use_llm_content_analysis'] = original_llm_setting
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)

    print("\n✅ LLM Content Analysis test completed!")


if __name__ == "__main__":
    asyncio.run(test_llm_content_analysis())
