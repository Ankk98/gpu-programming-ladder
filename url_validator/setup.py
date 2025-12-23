#!/usr/bin/env python3
"""
Setup script for GPU Programming Ladder URL Validation System
"""

import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8+")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def setup_virtual_environment():
    """Create and setup virtual environment."""
    if os.path.exists("gpu_ladder_env"):
        print("⚠️  Virtual environment already exists")
        return True

    commands = [
        "python3 -m venv gpu_ladder_env",
        "source gpu_ladder_env/bin/activate && pip install --upgrade pip"
    ]

    for cmd in commands:
        if not run_command(cmd, f"Setting up virtual environment ({cmd.split()[0]})"):
            return False

    return True


def install_dependencies():
    """Install Python dependencies."""
    cmd = "source gpu_ladder_env/bin/activate && pip install -r requirements.txt"
    return run_command(cmd, "Installing Python dependencies")


def check_lm_studio_connection():
    """Check if LM Studio is running and accessible."""
    import requests
    import time

    config = load_config()
    lm_studio_url = config['lm_studio']['url']

    print(f"🔍 Checking LM Studio connection at {lm_studio_url}...")

    try:
        # Try to connect with timeout
        response = requests.get(f"{lm_studio_url}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("✅ LM Studio connection successful")
            if 'data' in models and models['data']:
                model_names = [model['id'] for model in models['data']]
                print(f"   Available models: {', '.join(model_names)}")

                # Check if our preferred model is available
                preferred_model = config['lm_studio']['model']
                if any(preferred_model in model for model in model_names):
                    print(f"   ✅ Preferred model '{preferred_model}' is available")
                else:
                    print(f"   ⚠️  Preferred model '{preferred_model}' not found")
            return True
        else:
            print(f"❌ LM Studio returned status code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to LM Studio: {e}")
        print("   Please ensure LM Studio is running locally")
        return False


def load_config():
    """Load configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ config.json not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing config.json: {e}")
        return {}


def test_system():
    """Run basic system tests."""
    print("🧪 Running system tests...")

    # Test imports
    try:
        import sys
        sys.path.insert(0, '.')

        # Test basic imports
        import json
        import asyncio
        import aiohttp

        print("✅ Basic imports successful")

        # Test custom modules
        from url_extractor import extract_urls_from_data_js
        from task_creator_agent import TaskCreatorAgent

        print("✅ Custom module imports successful")

        # Test data extraction
        if os.path.exists("../data.js"):
            urls = extract_urls_from_data_js("../data.js")
            print(f"✅ URL extraction successful: {len(urls)} URLs found")
        else:
            print("⚠️  data.js not found - skipping URL extraction test")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 GPU Programming Ladder URL Validation System Setup")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        return False

    # Setup virtual environment
    if not setup_virtual_environment():
        return False

    # Install dependencies
    if not install_dependencies():
        return False

    # Test system
    if not test_system():
        return False

    # Check LM Studio connection (optional)
    print("\n🔍 Optional: Checking LM Studio connection...")
    lm_studio_ok = check_lm_studio_connection()
    if not lm_studio_ok:
        print("   ℹ️  LM Studio not detected. You can still run URL validation without AI replacements.")
        print("      Start LM Studio with GPT-4o model for full functionality.")

    # Setup complete
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Ensure LM Studio is running (optional but recommended)")
    print("   2. Run: source ../gpu_ladder_env/bin/activate")
    print("   3. Run: jupyter notebook url_validation_system.ipynb")
    print("   4. Follow the notebook instructions")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
