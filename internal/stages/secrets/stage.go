package secrets

import (
	"context"

	"github.com/drxc00/cloak/internal/pipeline"
)

// Stage detects high-entropy strings (API keys, tokens, passwords) using
// shannon-entropy analysis. It runs after the regex stage so deterministic
// patterns take priority.
type Stage struct{}

func NewStage() *Stage { return &Stage{} }

// Detect is a no-op placeholder pending entropy-scan implementation.
func (s *Stage) Detect(_ context.Context, _ string, _ []pipeline.Match) []pipeline.Match {
	return nil
}
