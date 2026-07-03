# Use official Python 3.9 slim image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app/

# Expose ports: 8000 for FastAPI, 8501 for Streamlit
EXPOSE 8000
EXPOSE 8501

# Copy and set execution permissions for the entrypoint script
RUN chmod +x /app/entrypoint.sh

# Run the startup script
ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
