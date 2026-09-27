# SupplySphere AI - Backend Image
# ------------------------------------------------------------
# Runs the FastAPI app (which also serves the frontend templates
# and static JS from /supplytemplates). MySQL runs as a separate
# service - see docker-compose.yml.
# ------------------------------------------------------------
FROM python:3.10-slim

WORKDIR /app

# System deps for mysql-connector / tensorflow wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        default-libmysqlclient-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the project
COPY backend/ ./backend/
COPY supplytemplates/ ./supplytemplates/
COPY ml/ ./ml/
COPY data/ ./data/
COPY database/ ./database/
COPY saved_models/ ./saved_models/
COPY outputs/ ./outputs/

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
