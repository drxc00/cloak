package ner

import (
	"context"

	"github.com/drxc00/cloak/internal/pipeline"
)

// Stage detects named entities (names, addresses, organisations) using an
// optional LLM fallback. It runs last so deterministic stages claim
// unambiguous ranges first.
type Stage struct{}

func NewStage() *Stage { return &Stage{} }

// Detect is a no-op placeholder pending LLM integration.
func (s *Stage) Detect(_ context.Context, _ string, _ []pipeline.Match) []pipeline.Match {
	return nil
}
