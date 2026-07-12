package secrets

import (
	"regexp"
	"testing"
)

func TestSetOf(t *testing.T) {
	s := setOf("a", "b", "c")
	if len(s) != 3 {
		t.Fatalf("expected 3 elements, got %d", len(s))
	}
	for _, k := range []string{"a", "b", "c"} {
		if !s[k] {
			t.Errorf("expected %q to be in set", k)
		}
	}
	if s["d"] {
		t.Error("expected 'd' to NOT be in set")
	}
}

func TestSetOf_Empty(t *testing.T) {
	s := setOf()
	if len(s) != 0 {
		t.Errorf("expected empty set, got %d elements", len(s))
	}
}

func TestContainsAny(t *testing.T) {
	tests := []struct {
		name     string
		s        string
		keywords []string
		want     bool
	}{
		{"exact match", "ghp_abc123", []string{"ghp_"}, true},
		{"case sensitive (lowercase input helps)", "GITHUB_TOKEN", []string{"github"}, false}, // not lowercased
		{"substring match", "prefix_ghp_suffix", []string{"ghp_"}, true},
		{"multiple keywords, one matches", "my_slack_token", []string{"ghp_", "slack", "xoxb"}, true},
		{"no match", "nothing here", []string{"ghp_", "xoxb"}, false},
		{"empty keywords", "anything", []string{}, false},
		{"empty input", "", []string{"ghp_"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := containsAny(tt.s, tt.keywords); got != tt.want {
				t.Errorf("containsAny(%q, %v) = %v, want %v", tt.s, tt.keywords, got, tt.want)
			}
		})
	}
}

func TestExtractSecret(t *testing.T) {
	text := "api_key=abc123secret"

	// Regex: api_key=(.+)  → group 1 is the secret
	re := regexp.MustCompile(`api_key=(.+)`)
	loc := re.FindStringSubmatchIndex(text)

	t.Run("secret group 0 (full match)", func(t *testing.T) {
		got, _, _ := extractSecret(text, loc, 0)
		want := "api_key=abc123secret"
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})

	t.Run("secret group 1 (capture group)", func(t *testing.T) {
		got, _, _ := extractSecret(text, loc, 1)
		want := "abc123secret"
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})

	t.Run("secret group out of range falls back to full match", func(t *testing.T) {
		got, _, _ := extractSecret(text, loc, 99)
		want := "api_key=abc123secret"
		if got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})
}

func TestExtractSecret_NoCaptureGroups(t *testing.T) {
	// Regex with no capture groups.
	re := regexp.MustCompile(`ghp_[0-9a-zA-Z]{36}`)
	text := "token=ghp_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz"
	loc := re.FindStringSubmatchIndex(text)

	got, _, _ := extractSecret(text, loc, 0)
	want := "ghp_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	// SecretGroup=1 with no capture groups should fall back to full match.
	got, _, _ = extractSecret(text, loc, 1)
	if got != want {
		t.Errorf("fallback: got %q, want %q", got, want)
	}
}

func TestOverlapsAny(t *testing.T) {
	spans := []span{{10, 20}, {30, 40}}

	tests := []struct {
		name  string
		start int
		end   int
		want  bool
	}{
		{"before all", 0, 5, false},
		{"between spans", 21, 29, false},
		{"after all", 41, 50, false},
		{"exact overlap first", 10, 20, true},
		{"partial overlap first", 5, 15, true},
		{"partial overlap second", 15, 25, true},
		{"partial overlap end", 35, 45, true},
		{"contains a span", 5, 45, true},
		{"adjacent (no gap)", 20, 30, false}, // [20,30) does NOT overlap [10,20) they touch
		{"empty input spans", 5, 10, false},  // test with nil/empty is separate
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := overlapsAny(tt.start, tt.end, spans); got != tt.want {
				t.Errorf("overlapsAny(%d, %d, %v) = %v, want %v", tt.start, tt.end, spans, got, tt.want)
			}
		})
	}
}

func TestOverlapsAny_EmptySpans(t *testing.T) {
	if overlapsAny(0, 10, nil) {
		t.Error("overlapsAny with nil spans should return false")
	}
	if overlapsAny(0, 10, []span{}) {
		t.Error("overlapsAny with empty spans should return false")
	}
}

func TestIsAllowed_ExactSecrets(t *testing.T) {
	a := &Allowlist{
		ExactSecrets: setOf("ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "test_placeholder_123456"),
	}

	if !isAllowed("ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", a) {
		t.Error("exact match should be allowed (suppressed)")
	}
	if isAllowed("ghp_real_token_123456789012345678901234567890", a) {
		t.Error("non-exact match should NOT be allowed (not suppressed)")
	}
	if isAllowed("", a) {
		t.Error("empty string should not match exact secrets")
	}
}

func TestIsAllowed_StopWords(t *testing.T) {
	a := &Allowlist{
		StopWords: setOf("password", "example", "test"),
	}

	tests := []struct {
		secret string
		want   bool // isAllowed = should be suppressed
	}{
		{"password", true},           // standalone word
		{"my password 123", true},     // contains standalone "password"
		{"my_password123", false},     // "password" embedded, not a word boundary
		{"example-key-123", true},     // "example" as whole word (surrounded by hyphens)
		{"examplekey", false},         // "example" embedded, not a word boundary
		{"real_secret_key_x9A2", false},
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.secret, func(t *testing.T) {
			if got := isAllowed(tt.secret, a); got != tt.want {
				t.Errorf("isAllowed(%q) = %v, want %v", tt.secret, got, tt.want)
			}
		})
	}
}

func TestIsAllowed_Regexes(t *testing.T) {
	a := &Allowlist{
		Regexes: []*regexp.Regexp{
			regexp.MustCompile(`^[a-zA-Z_.-]+$`), // purely alphabetic/punctuation
			regexp.MustCompile(`(?i)EXAMPLE$`),   // ends with "example"
		},
	}

	tests := []struct {
		secret string
		want   bool
	}{
		{"onlyletters", true},            // matches first regex
		{"only.letters.with.dots", true}, // matches first regex
		{"REALKEY_SUFFIX_EXAMPLE", true},   // matches second regex (case insensitive)
		{"REALKEY9", false},                  // doesn't match either
		{"abc123", false},                // has digit, no EXAMPLE suffix
	}
	for _, tt := range tests {
		t.Run(tt.secret, func(t *testing.T) {
			if got := isAllowed(tt.secret, a); got != tt.want {
				t.Errorf("isAllowed(%q) = %v, want %v", tt.secret, got, tt.want)
			}
		})
	}
}

func TestIsAllowed_NilAllowlist(t *testing.T) {
	// Should not panic.
	if isAllowed("anything", nil) {
		t.Error("nil allowlist should not suppress anything")
	}
}

func TestIsAllowed_Combined(t *testing.T) {
	// All three mechanisms active — exact should win.
	a := &Allowlist{
		ExactSecrets: setOf("exact_match"),
		StopWords:    setOf("stopword"),
		Regexes:      []*regexp.Regexp{regexp.MustCompile(`^\d+$`)},
	}

	if !isAllowed("exact_match", a) {
		t.Error("exact match should suppress")
	}
	if !isAllowed("stopword", a) {
		t.Error("standalone stopword should suppress")
	}
	if isAllowed("contains_stopword_here", a) {
		t.Error("embedded stopword should NOT suppress")
	}
	if !isAllowed("12345", a) {
		t.Error("regex should suppress all-digits")
	}
	if isAllowed("real_key_x9A2", a) {
		t.Error("real-looking key should pass through")
	}
}

func TestGenericAllowlist_RejectsAlphabetic(t *testing.T) {
	al := genericAllowlist()
	if !isAllowed("onlyletters", al) {
		t.Error("purely alphabetic should be suppressed")
	}
	if isAllowed("abc123def", al) {
		t.Error("alphanumeric should NOT be suppressed by alphabetic regex")
	}
}

func TestGenericAllowlist_StopWordsPresent(t *testing.T) {
	al := genericAllowlist()
	// Every stopword should suppress when it IS the secret (standalone).
	for w := range al.StopWords {
		if !isAllowed(w, al) {
			t.Errorf("standalone stopword %q should be suppressed", w)
		}
	}
	// Embedded stopwords (substring) should NOT suppress.
	if isAllowed("xpassword9", al) {
		t.Error("embedded password should NOT be suppressed")
	}
}
