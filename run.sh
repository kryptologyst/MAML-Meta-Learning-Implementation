#!/bin/bash
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
