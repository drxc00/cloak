package secrets

import (
	"context"
	"testing"

	"github.com/drxc00/cloak/internal/pipeline"
)

func TestStageDetect_VendorPrefix(t *testing.T) {
	s := NewStage()

	tests := []struct {
		name  string
		input string
		want  string // expected Match.Type
	}{
		{"github pat", "token=ghp_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz", "GITHUB_PAT"},
		{"github oauth", "gho_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz", "GITHUB_OAUTH"},
		{"gitlab", "glpat-abcdefghijklmnopqrstuvwxyz1234567890", "GITLAB_TOKEN"},
		{"aws key", "AKIALALEMEL33243OLIB", "AWS_ACCESS_KEY"},
		{"openai", "sk-1234567890abcdefghijT3BlbkFJ1234567890abcdefghij", "OPENAI_API_KEY"},
		{"slack bot", "xoxb-781236542736-2364535789652-GkwFDQoHqzXDVsC6GzqYUypD", "SLACK_BOT_TOKEN"},
		{"slack user", "xoxp-41684372915-1320496754-45609968301-e708ba56e1517a99f6b5fb07349476ef", "SLACK_USER_TOKEN"},
		{"slack webhook", "hooks.slack.com/services/T024TTTTT/BBB72BBL/AZAAA9u0pA4ad666eMgbi555", "SLACK_WEBHOOK_URL"},
		{"stripe", "sk_test_51Hc8n29fJd8sKdlA0293KfnaLc12345", "STRIPE_ACCESS_TOKEN"},
		{"generic", `api_key = "abcdefghijklmnopqrstuvwxyz0123456789"`, "GENERIC_API_KEY"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			matches := s.Detect(context.Background(), tt.input, nil)
			if len(matches) == 0 {
				t.Fatalf("expected at least 1 match, got 0")
			}
			if matches[0].Type != tt.want {
				t.Errorf("expected type %q, got %q", tt.want, matches[0].Type)
			}
		})
	}
}

func TestStageDetect_VendorBeforeGeneric(t *testing.T) {
	// When both a vendor rule and the generic rule could match the same
	// span, the vendor rule (processed first) should win.
	s := NewStage()
	input := `export GITHUB_TOKEN=ghp_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz`
	matches := s.Detect(context.Background(), input, nil)

	if len(matches) == 0 {
		t.Fatal("expected at least 1 match")
	}
	if matches[0].Type != "GITHUB_PAT" {
		t.Errorf("vendor rule should claim span first, got %q", matches[0].Type)
	}
}

func TestStageDetect_RespectsClaimed(t *testing.T) {
	// When the regex stage already claimed a span, secrets should skip it.
	s := NewStage()
	input := `export GITHUB_TOKEN=ghp_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz`

	// Simulate regex stage having already claimed the whole line.
	claimed := []pipeline.Match{
		{Type: "EMAIL", Start: 0, End: len(input), Text: input},
	}

	matches := s.Detect(context.Background(), input, claimed)
	if len(matches) != 0 {
		t.Errorf("expected 0 matches when span is already claimed, got %d", len(matches))
	}
}

func TestStageDetect_ExactSecretSuppressed(t *testing.T) {
	s := NewStage()
	// GitHub example placeholder should be suppressed by its ExactSecrets allowlist.
	matches := s.Detect(context.Background(), "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", nil)
	if len(matches) != 0 {
		t.Errorf("expected GitHub example key to be suppressed, got %d matches", len(matches))
	}
}

func TestStageDetect_StopWordSuppressed(t *testing.T) {
	s := NewStage()
	// Standalone stopword as the secret value should be suppressed.
	matches := s.Detect(context.Background(), `token = "example"`, nil)
	if len(matches) != 0 {
		t.Errorf("expected standalone stopword secret to be suppressed, got %d matches", len(matches))
	}
	// Embedded stopword (substring) should NOT be suppressed.
	matches = s.Detect(context.Background(), `token = "example1234567890"`, nil)
	if len(matches) == 0 {
		t.Error("expected embedded stopword to pass through")
	}
}

func TestStageDetect_LowEntropyRejected(t *testing.T) {
	s := NewStage()
	// A repeating pattern with low entropy should fail the entropy check.
	matches := s.Detect(context.Background(), `token = "aaaaaaaaaaaaaaaaaaaa"`, nil)
	if len(matches) != 0 {
		t.Errorf("expected low-entropy secret to be rejected, got %d matches", len(matches))
	}
}

func TestStageDetect_NoFalsePositiveOnNormalText(t *testing.T) {
	s := NewStage()
	normal := []string{
		"The service is running on port 8080",
		"database: postgres://localhost:5432/mydb",
		"Copyright 2024 Example Corp.",
		"const maxRetries = 5",
		"color: #ff0000",
		"version: 2.1.0",
	}

	for _, input := range normal {
		matches := s.Detect(context.Background(), input, nil)
		if len(matches) > 0 {
			t.Errorf("unexpected match on normal text %q: %+v", input, matches)
		}
	}
}

func TestStageDetect_MultipleMatchesOneLine(t *testing.T) {
	s := NewStage()
	input := `GH_TOKEN=ghp_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz and also slack: xoxb-781236542736-2364535789652-GkwFDQoHqzXDVsC6GzqYUypD`
	matches := s.Detect(context.Background(), input, nil)

	if len(matches) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(matches))
	}

	types := map[string]bool{}
	for _, m := range matches {
		types[m.Type] = true
	}
	if !types["GITHUB_PAT"] {
		t.Error("expected GITHUB_PAT match")
	}
	if !types["SLACK_BOT_TOKEN"] {
		t.Error("expected SLACK_BOT_TOKEN match")
	}
}

func TestStageDetect_EmptyInput(t *testing.T) {
	s := NewStage()
	matches := s.Detect(context.Background(), "", nil)
	if len(matches) != 0 {
		t.Errorf("expected 0 matches on empty input, got %d", len(matches))
	}
}
