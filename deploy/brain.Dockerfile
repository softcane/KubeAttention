# Brain Service Dockerfile
# Runs the Python gRPC server for node scoring

FROM python:3.12-slim

WORKDIR /app

ARG TORCH_VERSION=2.9.1+cpu
COPY brain/requirements.txt requirements.txt
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" && \
    pip install --no-cache-dir -r requirements.txt

COPY proto/ proto/
RUN mkdir -p gen/python && \
    python -m grpc_tools.protoc -I./proto --python_out=./gen/python --grpc_python_out=./gen/python ./proto/scheduler.proto

COPY brain/ brain/

RUN useradd --create-home --uid 1000 brain && \
    mkdir -p /var/run/kubeattention && \
    chown -R brain:brain /app /var/run/kubeattention

ENV PYTHONPATH=/app:/app/gen/python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 50051 8080

USER brain
CMD ["python", "-m", "brain.server"]
