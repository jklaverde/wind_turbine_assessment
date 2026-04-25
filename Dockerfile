FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/       src/
COPY pipeline.py .

# Phase 1 + Phase 2: wrangle data and train model at build time
RUN python pipeline.py

EXPOSE 8001

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "error"]
