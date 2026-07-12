package pipeline

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
)

type Kind int

const (
	KindString Kind = iota
	KindFile
	KindStdin
)

func (k Kind) String() string {
	switch k {
	case KindString:
		return "string"
	case KindFile:
		return "file"
	case KindStdin:
		return "stdin"
	default:
		return "unknown"
	}
}

type Input struct {
	kind   Kind
	name   string // filename, "-" for stdin, or "" for string input
	reader io.ReadCloser
}

func (in *Input) Kind() Kind   { return in.kind }
func (in *Input) Name() string { return in.name }

// Input itself satisfies io.ReadCloser, so it can be passed straight
// into io.ReadAll or a bufio.Scanner without unwrapping.
func (in *Input) Read(p []byte) (int, error) { return in.reader.Read(p) }
func (in *Input) Close() error               { return in.reader.Close() }

func FromString(s string) *Input {
	return &Input{
		kind:   KindString,
		name:   "",
		reader: io.NopCloser(strings.NewReader(s)),
	}
}

func FromFile(path string) (*Input, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	return &Input{
		kind:   KindFile,
		name:   path,
		reader: f,
	}, nil
}

func FromStdin() *Input {
	return &Input{kind: KindStdin, name: "-", reader: os.Stdin}
}

type Stage interface {
	Detect(ctx context.Context, text string, claimed []Match) []Match
}

type Pipeline struct {
	config *Config
	stages []Stage
}

// filters matches based on the pipeline's configuration, returning only those that are enabled.
func (p *Pipeline) filterByConfig(matches []Match) []Match {
	var filtered []Match
	for _, m := range matches {
		if p.config.EnabledTypes[m.Type] {
			filtered = append(filtered, m)
		}
	}
	return filtered
}

func New(config *Config, stages ...Stage) *Pipeline {
	return &Pipeline{config: config, stages: stages}
}

func (p *Pipeline) Run(ctx context.Context, input *Input) (*Result, error) {
	defer input.Close()

	data, err := io.ReadAll(input)
	if err != nil {
		return nil, fmt.Errorf("reading %s input %q: %w", input.Kind(), input.Name(), err)
	}
	text := string(data)

	var claimed []Match
	for _, stage := range p.stages {
		matches := stage.Detect(ctx, text, claimed)
		claimed = p.filterByConfig(mergeOverlaps(claimed, matches))
		if fullyCovered(text, claimed) {
			break
		}
	}

	return &Result{Input: input, Original: text, Redacted: applyRedactions(text, claimed), Matches: claimed}, nil
}

type Result struct {
	Input    *Input
	Original string
	Redacted string
	Matches  []Match
}

type Match struct {
	Type  string
	Start int
	End   int
	Text  string
}
