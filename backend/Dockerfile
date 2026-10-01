FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/data/model-cache OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
WORKDIR /app/backend
RUN groupadd --gid 10001 search && useradd --uid 10001 --gid search --create-home search
COPY backend/requirements.txt ./requirements.txt
# CPU wheels only; no CUDA runtime is installed.
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY --chown=search:search backend ./
COPY --chown=search:search README.md RESULTS.md /app/
RUN mkdir -p /app/data && chown -R search:search /app/data /app/backend /app
USER search
EXPOSE 8000
CMD ["sh", "-c", "python scripts/build_index.py && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --no-proxy-headers"]
