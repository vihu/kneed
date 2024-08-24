# Default
default:
    just --list --unsorted

# Build (dev)
build:
    cargo build

# Build (prod)
build-prod:
    cargo build --release

# Run clippy
clippy:
    cargo clippy -- -Dclippy::all -D warnings

# Test
test:
    cargo test --features testing
