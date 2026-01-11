# Contributing to KubeAttention

Thank you for your interest in contributing to KubeAttention! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)

## Code of Conduct

This project follows the [CNCF Code of Conduct](https://github.com/cncf/foundation/blob/main/code-of-conduct.md). Please be respectful and inclusive.

## Getting Started

### Prerequisites

- Go 1.22+
- Python 3.11+
- Docker
- Kind (for local testing)
- buf (for protobuf generation)

### Architecture Overview

```
KubeAttention/
├── proto/          # gRPC service definitions
├── brain/          # Python Transformer model (Scientist-Agent domain)
├── pkg/scheduler/  # Go K8s plugin (Gopher-Agent domain)
└── test/           # Test suites (Tester-Agent domain)
```

## Development Setup

```bash
# Clone the repository
git clone https://github.com/softcane/KubeAttention.git
cd KubeAttention

# Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r brain/requirements.txt

# Generate protobuf stubs
buf generate

# Build Go components
cd pkg/scheduler && go build ./...

# Run tests
./test/run_test_suite.sh
```

## Making Changes

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(brain): add temporal attention for time-series metrics
fix(scheduler): handle context cancellation properly
docs(readme): update installation instructions
```

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Make** your changes with tests
4. **Run** the audit gates:
   ```bash
   # Gate A: API deprecation
   pluto detect-files -d .
   
   # Gate B: Go linting
   golangci-lint run ./pkg/...
   
   # Gate C: Safety test
   ./test/gate_c_noop_safety.sh
   ```
5. **Submit** a pull request

### PR Requirements

- [ ] All tests pass
- [ ] No deprecated K8s APIs (Gate A)
- [ ] Go code follows idioms (Gate B)
- [ ] Safety test passes (Gate C)
- [ ] Documentation updated

## Coding Standards

### Go Code

- Use `context.Context` for cancellation (not `context.Background()` in hot paths)
- Use informer caches, not direct API calls
- Follow [Effective Go](https://golang.org/doc/effective_go)

```go
// ✅ Good: Uses informer cache
nodeInfo, _ := handle.SnapshotSharedLister().NodeInfos().Get(nodeName)

// ❌ Bad: Direct API call
node, _ := clientset.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
```

### Python Code

- Type hints required
- Use dataclasses for structured data
- Follow [PEP 8](https://peps.python.org/pep-0008/)

```python
# ✅ Good: Type hints + dataclass
@dataclass
class NodeMetrics:
    cpu_utilization: float
    memory_utilization: float

# ❌ Bad: No types
def process_metrics(data):
    return data["cpu"]
```

### Protocol Buffers

- All inter-service communication via `proto/scheduler.proto`
- Use snake_case for field names
- Add comments for each field

## Component Ownership

| Component | Owner Agent | Language |
|-----------|-------------|----------|
| `proto/` | Architect | Protobuf |
| `brain/` | Scientist | Python |
| `pkg/scheduler/` | Gopher | Go |
| `test/` | Tester | Bash/Python |

## Questions?

- Open a [GitHub Issue](https://github.com/softcane/KubeAttention/issues)
- Join our Slack: `#kubeattention`

---

Thank you for contributing! 🎉
