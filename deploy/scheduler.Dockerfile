FROM golang:1.25 AS builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/kubeattention-scheduler ./cmd/scheduler

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=builder /out/kubeattention-scheduler /usr/local/bin/kubeattention-scheduler
USER nonroot:nonroot
ENTRYPOINT ["/usr/local/bin/kubeattention-scheduler"]
