FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_XET=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY datagen ./datagen

# Download the base model and both adapters while building the image.
RUN python -c "from datagen.create_corpus import Specter2Encoder; Specter2Encoder(device='cpu')"

ENTRYPOINT ["python", "-m", "datagen.create_corpus"]
