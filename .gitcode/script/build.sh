#!/usr/bin/env bash
# Run selected Rust unit tests, then build pylance if all tests pass.
# Usage: .gitcode/script/build.sh [-- <test runner args>]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Library packages in the root Cargo workspace. Keep this list in sync with
# the workspace members in Cargo.toml; binary-only examples are not included.
CRATES=(
    #lance
    #lance-arrow
    lance-core
    #lance-datagen
    lance-encoding
    lance-file
    #lance-geo
    #lance-index
    #lance-io
    #lance-linalg
    #lance-namespace
    #lance-namespace-impls
    #lance-namespace-datafusion
    #lance-select
    #lance-tokenizer
    #lance-table
    #lance-derive
    #lance-test-macros
    #lance-testing
    #lance-tools
    #fsst
    #lance-bitpacking
    #lance-arrow-scalar
    #lance-arrow-stats
)

if [ "${#CRATES[@]}" -eq 0 ]; then
    echo "ERROR: CRATES is empty; select at least one crate." >&2
    exit 1
fi

TEST_RUNNER_ARGS=()
if [ "$#" -gt 0 ]; then
    if [ "$1" != "--" ]; then
        echo "Usage: $0 [-- <test runner args>]" >&2
        exit 2
    fi
    shift
    TEST_RUNNER_ARGS=("$@")
fi

cd "$REPO_ROOT"
echo "==> Running unit tests for: ${CRATES[*]}"
for crate in "${CRATES[@]}"; do
    echo "==> cargo test -p $crate --lib ${TEST_RUNNER_ARGS[*]}"
    cargo test -p "$crate" --lib -- "${TEST_RUNNER_ARGS[@]}"
done

echo "==> All unit tests passed, building pylance..."
cd "$REPO_ROOT/python"
maturin build --release

echo "==> Done."

