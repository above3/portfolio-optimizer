# Containerised deployment for the Portfolio Optimizer (Streamlit).
# Build:  docker build -t portfolio-optimizer .
# Run:    docker run -p 8501:8501 portfolio-optimizer
FROM python:3.11-slim

# System deps: curl for the health-check; build-essential for any wheels that
# need compilation (PyPortfolioOpt -> cvxpy / osqp).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application source.
COPY . .

# Persist the local SQLite state outside the image layer when a volume is mounted.
ENV PORTFOLIO_DB_PATH=/app/data/.portfolio_state.db
RUN mkdir -p /app/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
