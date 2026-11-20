# ---- build stage: resolve and install dependencies into a venv -------------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- runtime stage: only the venv + application code -----------------------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    DATABASE_URL=sqlite:////data/app.db \
    LITELLM_LOCAL_MODEL_COST_MAP=True

RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app:app /data
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY app ./app

USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)" || exit 1

# PORT is honoured so the same image runs on Render / Railway / Fly.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
