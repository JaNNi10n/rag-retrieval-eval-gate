# CPU-only image for the /search API. The reranker arm is off by default (it was 6-14x slower than hybrid at p50 even on GPU).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    RAGBENCH_DATA_DIR=/home/app/data \
    RAGBENCH_INDEX_DIR=/home/app/indexes \
    RAGBENCH_DEVICE=cpu \
    RAGBENCH_ARMS=bm25,dense,hybrid

WORKDIR /app

COPY requirements.txt .
RUN pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY ragbench/ ragbench/
# Frozen RRF k chosen on the train split; the API reads it at startup.
COPY results/rrf_tuning.json results/rrf_tuning.json

RUN useradd --create-home --uid 10001 app && mkdir -p /home/app/data /home/app/indexes && chown -R app:app /home/app
USER app

EXPOSE 8000
# First start downloads SciFact + bge-small and encodes 5,183 docs on CPU. That is slow (about 1 doc/s measured on
# the author's busy laptop CPU, results/cpu_encode_probe.log), so mount /home/app to keep the index between runs.
CMD ["uvicorn", "ragbench.api:app", "--host", "0.0.0.0", "--port", "8000"]
