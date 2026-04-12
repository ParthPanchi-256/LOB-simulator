FROM python:3.10-slim

# Set up working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire environment package into the container first
# so that pip install . can find the 'server' subdirectory defined in pyproject.toml
COPY . .

# Install python dependencies
RUN pip install --no-cache-dir .
RUN pip install --no-cache-dir uvicorn

# Run the FastAPI app via Uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
