#!/bin/bash
# Script to start BSmart Python Backend Server in WSL/Linux

ENV_PATH="./glass/glass/bin/activate"
if [ -f "$ENV_PATH" ]; then
    echo "Activating virtual environment from $ENV_PATH..."
    source "$ENV_PATH"
elif [ -f "./glass/bin/activate" ]; then
    echo "Activating virtual environment from ./glass/bin/activate..."
    source "./glass/bin/activate"
fi

echo "Starting BSmart FastAPI Backend Server on port 8000..."
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
