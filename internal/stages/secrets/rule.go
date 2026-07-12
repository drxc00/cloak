package secrets

import "regexp"

// Rule defines a single secret-detection pattern: a regex, an optional
// entropy floor, an optional keyword-based pre-filter, and an allowlist
// to suppress known false positives.
type Rule struct {
	Type        string
	Keywords    []string // lowercase pre-filter terms; empty = always run (generic rule)
	Regex       *regexp.Regexp
	SecretGroup int     // capture-group index holding the actual secret value
	MinEntropy  float64 // 0 disables the entropy check for this rule
	Allowlist   *Allowlist
}

// Allowlist suppresses false positives at the per-finding level.
type Allowlist struct {
	Regexes      []*regexp.Regexp
	StopWords    map[string]bool
	ExactSecrets map[string]bool
}

// Candidate is a pre-confidence-scored potential secret, kept separate from
// pipeline.Match so entropy/allowlist reasoning is inspectable in tests and
// dry-run reports.
type Candidate struct {
	Value    string
	Start    int
	End      int
	RuleType string
	Entropy  float64
	Signals  []string // "prefix", "keyword", "entropy", "basicauth"
}
