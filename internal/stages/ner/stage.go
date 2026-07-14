package ner

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/drxc00/cloak/internal/pipeline"
)

// The token-classification model tags address components (street, city,
// state/postcode) as separate spans because it decodes per-token, not per
// entity. connectorRe matches the short "glue" between two such spans
// (", ", ", IL ", ", CA ") — a comma and/or a bare 1-3 char code — so they
// can be merged into the single contiguous ADDRESS the benchmark expects.
var connectorRe = regexp.MustCompile(`^,?\s*[A-Z0-9]{0,6}\s*,?\s*$`)

// houseNumberRe matches a bare street number ("742", "221B") immediately
// before a merged ADDRESS span, which the model doesn't tag on its own.
var houseNumberRe = regexp.MustCompile(`(?:^|\s)(\d{1,6}[A-Za-z]?)\s*$`)

// ---------------------------------------------------------------------------
// Sidecar protocol types
// ---------------------------------------------------------------------------

type request struct {
	Text   string   `json:"text"`
	Labels []string `json:"labels"`
}

type response struct {
	Entities []entity `json:"entities"`
}

type entity struct {
	Start int     `json:"start"`
	End   int     `json:"end"`
	Type  string  `json:"type"`
	Score float64 `json:"score"`
}

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

// Stage detects named entities (names, addresses, usernames) via the
// cloak-nerd sidecar binary, which runs a supervised token-classification
// ONNX model. cloak-nerd loads the model once and serves NDJSON requests
// over stdin/stdout for its whole lifetime, so Stage keeps one subprocess
// running for as long as the Stage itself is used, rather than spawning
// (and reloading the model) per request.
type Stage struct {
	binPath       string
	modelPath     string
	tokenizerPath string
	configPath    string

	mu     sync.Mutex
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
}

func NewStage() *Stage {
	// cloak-nerd is installed by `cloak init` into the cache dir.
	cacheDir, err := userCacheDir()
	if err != nil {
		return nil
	}
	binPath := filepath.Join(cacheDir, "bin", "cloak-nerd")
	if _, err := os.Stat(binPath); err != nil {
		return nil // not installed
	}

	modelPath := filepath.Join(cacheDir, "models", "model.onnx")
	tokPath := filepath.Join(cacheDir, "models", "tokenizer.json")
	configPath := filepath.Join(cacheDir, "models", "model_config.json")

	return &Stage{
		binPath:       binPath,
		modelPath:     modelPath,
		tokenizerPath: tokPath,
		configPath:    configPath,
	}
}

// Detect runs NER on the remaining unclaimed text using cloak-nerd.
//
// It collects text regions that haven't been claimed by earlier stages,
// sends them to the sidecar, and maps global byte offsets for returned
// entities. If cloak-nerd isn't installed or fails, it returns nil
// (graceful degradation).
func (s *Stage) Detect(_ context.Context, text string, claimed []pipeline.Match) []pipeline.Match {
	if s == nil {
		return nil
	}

	// Collect unclaimed text regions and their global offsets.
	type chunk struct {
		text   string
		offset int // global byte offset into the original text
	}
	var chunks []chunk

	// Sort claimed matches by start position.
	sorted := make([]pipeline.Match, len(claimed))
	copy(sorted, claimed)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Start < sorted[j].Start })

	// Extract gaps between claimed spans.
	pos := 0
	for _, m := range sorted {
		if m.Start > pos {
			raw := text[pos:m.Start]
			gap := strings.TrimSpace(raw)
			if len(gap) > 0 {
				chunks = append(chunks, chunk{text: gap, offset: pos + strings.Index(raw, gap)})
			}
		}
		if m.End > pos {
			pos = m.End
		}
	}
	// Trailing gap after last claim.
	if pos < len(text) {
		raw := text[pos:]
		gap := strings.TrimSpace(raw)
		if len(gap) > 0 {
			chunks = append(chunks, chunk{text: gap, offset: pos + strings.Index(raw, gap)})
		}
	}

	if len(chunks) == 0 {
		return nil
	}

	// Core-3 label set — the classifier only knows NAME, ADDRESS, USERNAME.
	// These are passed as an allow-list filter to the sidecar.
	labels := []string{"NAME", "ADDRESS", "USERNAME"}

	var matches []pipeline.Match

	for _, ch := range chunks {
		entities, err := s.runSidecar(ch.text, labels)
		if err != nil {
			// Graceful degradation — if sidecar fails, skip NER.
			continue
		}
		for _, ent := range entities {
			matches = append(matches, pipeline.Match{
				Type:  ent.Type,
				Start: ch.offset + ent.Start,
				End:   ch.offset + ent.End,
				Text:  ch.text[ent.Start:ent.End],
			})
		}
	}

	sort.Slice(matches, func(i, j int) bool { return matches[i].Start < matches[j].Start })
	return mergeAdjacentSpans(text, matches)
}

// mergeAdjacentSpans collapses consecutive same-type matches that the model
// split at token boundaries (see connectorRe/houseNumberRe above) back into
// the single entity a human would redact as one unit.
func mergeAdjacentSpans(text string, matches []pipeline.Match) []pipeline.Match {
	if len(matches) == 0 {
		return matches
	}

	merged := make([]pipeline.Match, 0, len(matches))
	cur := matches[0]
	for _, next := range matches[1:] {
		if next.Type == cur.Type && connectorRe.MatchString(text[cur.End:next.Start]) {
			cur.End = next.End
			cur.Text = text[cur.Start:cur.End]
			continue
		}
		merged = append(merged, cur)
		cur = next
	}
	merged = append(merged, cur)

	for i := range merged {
		if merged[i].Type != "ADDRESS" {
			continue
		}
		if loc := houseNumberRe.FindStringSubmatchIndex(text[:merged[i].Start]); loc != nil {
			merged[i].Start = loc[2]
			merged[i].Text = text[merged[i].Start:merged[i].End]
		}
	}

	return merged
}

// start launches the long-lived cloak-nerd subprocess and wires up its
// stdin/stdout pipes. Must be called with s.mu held.
func (s *Stage) start() error {
	cmd := exec.Command(s.binPath,
		"--model", s.modelPath,
		"--tokenizer", s.tokenizerPath,
		"--config", s.configPath,
		"--threshold", "0.5",
	)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start cloak-nerd: %w", err)
	}

	s.cmd = cmd
	s.stdin = stdin
	s.stdout = bufio.NewReader(stdout)
	return nil
}

// runSidecar sends a single text to the (already-running, or lazily
// started) cloak-nerd subprocess and parses its response.
func (s *Stage) runSidecar(text string, labels []string) ([]entity, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cmd == nil {
		if err := s.start(); err != nil {
			return nil, err
		}
	}

	req := request{Text: text, Labels: labels}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	if _, err := s.stdin.Write(append(reqJSON, '\n')); err != nil {
		return nil, fmt.Errorf("write to cloak-nerd: %w", err)
	}

	line, err := s.stdout.ReadBytes('\n')
	if err != nil {
		return nil, fmt.Errorf("read from cloak-nerd: %w", err)
	}

	var resp response
	if err := json.Unmarshal(line, &resp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	return resp.Entities, nil
}

// Close stops the cloak-nerd subprocess, if running. Safe to call on a
// nil Stage or one that was never used.
func (s *Stage) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.cmd == nil {
		return nil
	}
	_ = s.stdin.Close()
	err := s.cmd.Wait()
	s.cmd = nil
	return err
}

// ---------------------------------------------------------------------------
// Cache directory
// ---------------------------------------------------------------------------

func userCacheDir() (string, error) {
	dir, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("user cache dir: %w", err)
	}
	return filepath.Join(dir, "cloak"), nil
}

// Installed reports whether cloak-nerd is available on disk.
func Installed() bool {
	cacheDir, err := userCacheDir()
	if err != nil {
		return false
	}
	binPath := filepath.Join(cacheDir, "bin", "cloak-nerd")
	_, err = os.Stat(binPath)
	return err == nil
}
