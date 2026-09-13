FROM golang:1.25-alpine AS builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/collector ./cmd/collector

FROM alpine:3.22
RUN addgroup -g 1000 collector && adduser -D -u 1000 -G collector collector
WORKDIR /app
COPY --from=builder /out/collector /app/collector
USER collector
ENTRYPOINT ["/app/collector"]
