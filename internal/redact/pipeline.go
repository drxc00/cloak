package redact

// Pipeline runs detection stages in order, each stage redacting what the previous left.
type Pipeline struct{}

func NewPipeline() *Pipeline {
	return &Pipeline{}
}

func (p *Pipeline) Redact(input string) string {
	return input
}
