# Cloak Test Data

Static test fixtures for validating cloak's detection stages and
benchmarking redaction throughput.

## Usage

```bash
# Redact a single file
cloak redact testdata/config/setup-env.sh

# Redact all test fixtures
find testdata -type f -name '*.sh' -o -name '*.yml' -o -name '*.tf' \
  -o -name '*.log' -o -name '*.md' | while read f; do
  cloak redact "$f" > "${f}.redacted"
done

# Run integration benchmark
go test ./integration/ -v -run TestBenchmarkCases
```

## Disclosure

Some test data files in this directory were generated with the assistance of
large language models. All generated content has been reviewed for accuracy
and contains no real, live credentials. Every token and key shown is a
placeholder or synthetic value created for testing purposes only.
