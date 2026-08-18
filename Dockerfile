# Small on purpose: the whole argument in DESIGN.md is that you do not need a
# headless browser to do this job, and the image is where that claim shows up.
# ~180MB, versus ~900MB for anything carrying Chromium or torch.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# lxml wheels cover manylinux, so no build toolchain is needed. curl is here
# only for the container healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY DESIGN.md DECISIONS.md README.md ./

RUN mkdir -p /app/data && useradd -m -u 10001 sentinel && chown -R sentinel /app
USER sentinel

ENV SENTINEL_DB=/app/data/sentinel.db \
    PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

# One worker, deliberately. The scheduler is a singleton per process -- two
# workers would mean two schedulers polling the same sources twice, which is
# both wasteful and exactly the traffic pattern the pacing layer exists to
# avoid. Scaling out means moving the scheduler to its own process, not adding
# workers here.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
