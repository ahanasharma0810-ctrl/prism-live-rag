# CONFIGURATION ONLY - this image has not been built or run during Phase 1
# (Docker is not available in the build environment). See reports/PHASE_1_REPORT.md.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY . .

# Indexes are built from data/corpus/ at container start, then a sample query is answered.
CMD ["sh", "-c", "python -m src.corpus.build_index && python -m src.baseline --query \"$SAMPLE_QUERY\""]
