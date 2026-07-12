package regex

import (
	"context"
	"sort"

	"github.com/drxc00/cloak/internal/pipeline"
)

// Stage detects sensitive patterns using compiled regular expressions.
// It is the first (and fastest) pipeline stage, handling unambiguous
// formats like email addresses, IPs, SSNs, credit cards, JWTs, and
// PEM-encoded keys.
type Stage struct{}

func NewStage() *Stage { return &Stage{} }

// Detect scans text for every registered pattern, returning non-overlapping
// matches ordered by priority (higher-priority patterns claim their ranges
// first).
func (s *Stage) Detect(_ context.Context, text string, _ []pipeline.Match) []pipeline.Match {
	if len(Patterns) == 0 {
		return nil
	}

	// Work on a copy sorted by descending priority so higher-priority
	// patterns claim their ranges first.
	sortedPatterns := make([]Pattern, len(Patterns))
	copy(sortedPatterns, Patterns)
	sort.Slice(sortedPatterns, func(i, j int) bool {
		return sortedPatterns[i].Priority > sortedPatterns[j].Priority
	})

	// Track occupied byte ranges.
	occupied := make([]range2, 0)

	var matches []pipeline.Match

	for _, pat := range sortedPatterns {
		locs := pat.Regex.FindAllStringIndex(text, -1)
		for _, loc := range locs {
			start, end := loc[0], loc[1]

			// Skip if already claimed by a higher-priority pattern.
			if overlapsAny(start, end, occupied) {
				continue
			}

			hit := text[start:end]

			// Run optional validator.
			if pat.Validate != nil && !pat.Validate(hit) {
				continue
			}

			occupied = append(occupied, range2{start, end})
			matches = append(matches, pipeline.Match{
				Type:  pat.Type,
				Start: start,
				End:   end,
				Text:  hit,
			})
		}
	}

	// Return in document order.
	sort.Slice(matches, func(i, j int) bool { return matches[i].Start < matches[j].Start })
	return matches
}

type range2 struct{ start, end int }

// checks if the given range overlaps with any of the provided ranges.
func overlapsAny(start, end int, spans []range2) bool {
	for _, s := range spans {
		if start < s.end && s.start < end {
			return true
		}
	}
	return false
}
