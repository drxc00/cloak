package redact_test

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/drxc00/cloak/internal/redact"
)

type benchmarkCase struct {
	ID             int    `json:"id"`
	Category       string `json:"category"`
	Input          string `json:"input"`
	ExpectedOutput string `json:"expected_output"`
	Notes          string `json:"notes"`
}

func loadCases(t *testing.T) []benchmarkCase {
	t.Helper()
	path := filepath.Join("..", "..", "testdata", "benchmark_cases.jsonl")
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open benchmark_cases.jsonl: %v", err)
	}
	defer f.Close()

	var cases []benchmarkCase
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var c benchmarkCase
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			t.Fatalf("parse line: %v", err)
		}
		cases = append(cases, c)
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan: %v", err)
	}
	return cases
}

func TestBenchmarkCases(t *testing.T) {
	cases := loadCases(t)
	p := redact.NewPipeline()

	pass, fail := 0, 0
	for _, tc := range cases {
		tc := tc
		t.Run(fmt.Sprintf("%02d_%s", tc.ID, tc.Category), func(t *testing.T) {
			got := p.Redact(tc.Input)
			if got == tc.ExpectedOutput {
				pass++
				return
			}
			fail++
			t.Errorf(
				"\nInput:    %s\nExpected: %s\nGot:      %s\nNotes:    %s",
				tc.Input, tc.ExpectedOutput, got, tc.Notes,
			)
		})
	}
}

// BenchmarkRedact measures pipeline throughput across all cases.
func BenchmarkRedact(b *testing.B) {
	path := filepath.Join("..", "..", "testdata", "benchmark_cases.jsonl")
	f, err := os.Open(path)
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	defer f.Close()

	var cases []benchmarkCase
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var c benchmarkCase
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			b.Fatalf("parse: %v", err)
		}
		cases = append(cases, c)
	}

	p := redact.NewPipeline()

	for b.Loop() {
		for _, tc := range cases {
			p.Redact(tc.Input)
		}
	}
}
