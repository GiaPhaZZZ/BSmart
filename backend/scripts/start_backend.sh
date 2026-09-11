#!/bin/bash
# Script to start BSmart Python Backend Server in WSL/Linux
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BACKEND_DIR/.." && pwd)"

cd "$BACKEND_DIR"

if [ -f "$REPO_ROOT/glass/glass/bin/activate" ]; then
    echo "Activating virtual environment from $REPO_ROOT/glass/glass/bin/activate..."
    source "$REPO_ROOT/glass/glass/bin/activate"
elif [ -f "$REPO_ROOT/glass/bin/activate" ]; then
    echo "Activating virtual environment from $REPO_ROOT/glass/bin/activate..."
    source "$REPO_ROOT/glass/bin/activate"
fi

echo "Starting BSmart FastAPI Backend Server on port 8000..."
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
