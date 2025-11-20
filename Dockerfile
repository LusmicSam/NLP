FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface
COPY . .
CMD ["python", "benchmark.py"]
ENV HUGGING_FACE_HUB_TOKEN="USE HUGGINF FACE TOKKEN ON READ MODE HERE"
