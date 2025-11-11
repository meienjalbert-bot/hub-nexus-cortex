# apps/orchestrator/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Outils utiles pour healthchecks/debug (wget, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 1) Dépendances Python
COPY apps/orchestrator/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 2) Code applicatif (API)
COPY apps/orchestrator/ /app/

# 3) Modules "core" (MoME, consensus, etc.) -> C'EST LE POINT QUI MANQUAIT
COPY core/ /app/core/

# 4) (Optionnel) configs .env montées par docker-compose via env_file

ENV PYTHONUNBUFFERED=1
EXPOSE 8100

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100"]

