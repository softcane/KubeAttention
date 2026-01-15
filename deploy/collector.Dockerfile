# Collector Service Dockerfile
FROM golang:alpine AS builder

WORKDIR /app

# Install git for dependencies
RUN apk add --no-cache git

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build collector
RUN go build -o collector ./cmd/collector/main.go

# Final stage
FROM alpine:3.19
WORKDIR /app
COPY --from=builder /app/collector /app/collector
ENTRYPOINT ["/app/collector"]
