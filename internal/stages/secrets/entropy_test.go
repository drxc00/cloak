package secrets

import (
	"math"
	"testing"
)

func TestShannonEntropy(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    float64
		tol     float64 // absolute tolerance
	}{
		{"empty string", "", 0, 0},
		{"single char", "a", 0, 0.001},
		{"two same chars", "aa", 0, 0.001},
		{"uniform hex", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", 0, 0.001},
		{"one bit per char (binary)", "01010101010101010101010101010101", 1.0, 0.001},
		{"two bits per char (hex)", "0123456789abcdef0123456789abcdef", 4.0, 0.001},
		{"alphanumeric mix", "abc123", 2.585, 0.001},
		{"english word low entropy", "password", 2.75, 0.001},
		{"base64-ish", "dGhpcyBpcyBhIHRlc3Qgc3RyaW5nIHRoYXQgaXMgcXVpdGUgbG9uZw==", 4.711, 0.001},
		{"random alphanum 40chars", "a7Kx9mQw3NpR5tYv2BcF8hJ1LzX4dG6W", 5.0, 0.001},
		{"UUID", "550e8400-e29b-41d4-a716-446655440000", 3.391, 0.001},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shannonEntropy(tt.input)
			if math.Abs(got-tt.want) > tt.tol {
				t.Errorf("shannonEntropy(%q) = %v, want %v (±%v)", tt.input, got, tt.want, tt.tol)
			}
		})
	}
}

func TestShannonEntropy_Monotonic(t *testing.T) {
	// Longer random strings should have higher or equal entropy than shorter ones
	// built from the same alphabet.
	short := shannonEntropy("a7Kx9mQw3N")
	long := shannonEntropy("a7Kx9mQw3NpR5tYv2BcF8hJ1LzX4dG6W")
	if long < short {
		t.Errorf("longer random string entropy (%v) should be >= shorter (%v)", long, short)
	}
}

func TestShannonEntropy_Threshold(t *testing.T) {
	// Realistic secrets should clear typical thresholds.
	highEntropy := []string{
		"ghp_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz",
		"sk-1234567890abcdefghijT3BlbkFJ1234567890abcdefghij",
		"xoxb-781236542736-2364535789652-GkwFDQoHqzXDVsC6GzqYUypD",
		"AKIAIOSFODNN7EXAMPL",
	}
	for _, s := range highEntropy {
		e := shannonEntropy(s)
		if e < 3.0 {
			t.Errorf("expected entropy ≥ 3.0 for %q, got %v", s, e)
		}
	}

	// Clearly non-secret strings should have low entropy.
	lowEntropy := []string{
		"password",
		"example",
	}
	for _, s := range lowEntropy {
		e := shannonEntropy(s)
		if e > 3.0 {
			t.Errorf("expected entropy ≤ 3.0 for %q, got %v", s, e)
		}
	}
}
