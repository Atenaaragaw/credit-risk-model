# Use a Python base image suitable for production
FROM python:3.10-slim

# Set environment variables for non-buffered output and work directory
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy only the requirements file first to leverage Docker's build cache
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y libgomp1 && \
    pip --timeout 300 install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ src/
COPY data/ data/
COPY model model
# Copy MLflow tracking files/db (needed for tracking history, no longer for loading)
COPY mlflow.db .
COPY mlruns/ mlruns/

# Expose the port uvicorn will run on
EXPOSE 8000

# Command to run the application using uvicorn (exec form for graceful shutdown)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]