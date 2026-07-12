package redact

import (
	"context"

	"github.com/drxc00/cloak/internal/pipeline"
	"github.com/drxc00/cloak/internal/stages/ner"
	"github.com/drxc00/cloak/internal/stages/regex"
	"github.com/drxc00/cloak/internal/stages/secrets"
)

// Pipeline wraps the core pipeline with a simple Redact API.
type Pipeline struct {
	inner *pipeline.Pipeline
}

// NewPipeline builds a pipeline with the default config and all three stages
// (regex → secrets → ner).
func NewPipeline() *Pipeline {
	config := pipeline.NewConfig()
	p := pipeline.New(
		config,
		regex.NewStage(),
		secrets.NewStage(),
		ner.NewStage(),
	)
	return &Pipeline{inner: p}
}

// Redact scans input and replaces every detected sensitive span with a
// "[REDACTED - TYPE]" marker.
func (p *Pipeline) Redact(input string) string {
	result, err := p.inner.Run(context.Background(), pipeline.FromString(input))
	if err != nil {
		return input
	}
	return result.Redacted
}
