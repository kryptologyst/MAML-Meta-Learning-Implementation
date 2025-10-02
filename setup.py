#!/usr/bin/env python3
"""
Setup script for MAML Meta-Learning Implementation
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def setup_project():
    """Setup the MAML project"""
    print("🚀 Setting up MAML Meta-Learning Implementation")
    print("=" * 50)
    
    # Create necessary directories
    directories = ["checkpoints", "logs", "tests", "docs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("⚠️  Some dependencies may have failed to install. Please check manually.")
    
    # Run tests
    if Path("test_maml.py").exists():
        if run_command("python -m pytest test_maml.py -v", "Running tests"):
            print("✅ All tests passed!")
        else:
            print("⚠️  Some tests failed. Please check the implementation.")
    
    # Check code formatting
    if run_command("python -m black --check 0151.py", "Checking code formatting"):
        print("✅ Code formatting is correct")
    else:
        print("⚠️  Code formatting issues detected. Run 'black 0151.py' to fix.")
    
    # Create a simple run script
    run_script = """#!/bin/bash
# MAML Meta-Learning Run Script

echo "🧠 MAML Meta-Learning Implementation"
echo "=================================="

if [ "$1" = "ui" ]; then
    echo "🚀 Starting Streamlit UI..."
    streamlit run 0151.py streamlit
elif [ "$1" = "train" ]; then
    echo "🏋️ Starting training..."
    python 0151.py
elif [ "$1" = "test" ]; then
    echo "🧪 Running tests..."
    python -m pytest test_maml.py -v
else
    echo "Usage: $0 [ui|train|test]"
    echo "  ui    - Start the Streamlit web interface"
    echo "  train - Run training from command line"
    echo "  test  - Run the test suite"
fi
"""
    
    with open("run.sh", "w") as f:
        f.write(run_script)
    
    # Make run script executable
    os.chmod("run.sh", 0o755)
    print("📝 Created run script: run.sh")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Run './run.sh ui' to start the web interface")
    print("2. Run './run.sh train' to train from command line")
    print("3. Run './run.sh test' to run tests")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    setup_project()
