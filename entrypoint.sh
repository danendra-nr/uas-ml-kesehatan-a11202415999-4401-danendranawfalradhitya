#!/bin/sh

# Start FastAPI backend service in the background
echo "Starting FastAPI backend service on port 8000..."
python api_fastapi.py &

# Start Streamlit dashboard in the foreground
echo "Starting Streamlit dashboard on port 8501..."
streamlit run app_streamlit.py --server.port=8501 --server.address=0.0.0.0
