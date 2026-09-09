# Two images from one file.
#
#   target "base" -> the web tier: FastAPI, the console, no ML stack at all.
#   target "tool" -> the equipment tier: adds NumPy, CPU PyTorch and the
#                    checkpoint, because only this process runs the classifier.
#
# Splitting them keeps the web image small and means scaling the web tier does
# not multiply a 1.5 GB PyTorch image.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-services.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-services.txt

COPY backend ./backend
COPY frontend ./frontend
COPY ml/__init__.py ml/labels.py ml/model.py ./ml/

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM base AS tool

# NumPy is a hard requirement of the detector. torch does NOT pull it in, and
# without it CNNDetector cannot build its input tensor - the first build of this
# image fell back to the heuristic for exactly that reason.
RUN pip install --no-cache-dir "numpy>=1.26"

# CPU wheels only - this container classifies a handful of wafers per minute,
# so a CUDA image would be several gigabytes for no benefit.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY ml/checkpoints ./ml/checkpoints

CMD ["python", "-m", "backend.tool_service"]
