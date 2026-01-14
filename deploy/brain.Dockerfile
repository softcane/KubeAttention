# Brain Service Dockerfile
# Runs the Python gRPC server for node scoring

FROM python:3.11-slim

WORKDIR /app

# Install dependencies with pinned protobuf version that supports runtime_version
COPY brain/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir protobuf>=5.27.0 grpcio-tools>=1.66.0

# Copy proto files first
COPY proto/ proto/

# Create gen directory and regenerate stubs with matching protobuf version
RUN mkdir -p gen/python && \
    python -m grpc_tools.protoc -I./proto --python_out=./gen/python --grpc_python_out=./gen/python ./proto/scheduler.proto

# Copy brain module
COPY brain/ brain/

# Create UDS socket directory
RUN mkdir -p /var/run/kubeattention

# Set Python path
ENV PYTHONPATH=/app:/app/gen/python

# Expose gRPC and metrics ports
EXPOSE 50051 9090

# Run server
CMD ["python", "-m", "brain.server"]
