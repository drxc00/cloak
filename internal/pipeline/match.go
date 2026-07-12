package pipeline

import (
	"sort"
	"strings"
)

// mergeOverlaps combines already-claimed matches with new matches from the
// current stage. Earlier stages (already claimed) take priority
// any new match that overlaps with an existing claim is dropped.
func mergeOverlaps(claimed, incoming []Match) []Match {
	if len(incoming) == 0 {
		return claimed
	}

	// Sort incoming by start position for stable processing
	sort.Slice(incoming, func(i, j int) bool { return incoming[i].Start < incoming[j].Start })

	result := make([]Match, 0, len(claimed)+len(incoming))
	result = append(result, claimed...)

	for _, m := range incoming {
		if overlapsAny(m, claimed) {
			continue
		}
		result = append(result, m)
	}

	// Keep the full set sorted.
	sort.Slice(result, func(i, j int) bool { return result[i].Start < result[j].Start })
	return result
}

func overlapsAny(m Match, existing []Match) bool {
	for _, e := range existing {
		if overlaps(m, e) {
			return true
		}
	}
	return false
}

// overlaps checks whether two matches share any byte range.
func overlaps(a, b Match) bool {
	return a.Start < b.End && b.Start < a.End
}

// fullyCovered returns true when every byte of text falls inside at least one match.
// Therefore, once everything is claimed, later stages can be skipped.
func fullyCovered(text string, matches []Match) bool {
	if len(matches) == 0 {
		return false
	}

	// Sort a copy so we don't mutate the caller's slice.
	sorted := make([]Match, len(matches))
	copy(sorted, matches)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Start < sorted[j].Start })

	pos := 0
	for _, m := range sorted {
		if m.Start > pos {
			return false // gap before this match
		}
		if m.End > pos {
			pos = m.End
		}
	}
	return pos >= len(text)
}

// applyRedactions replaces every matched region in text with a
// "[REDACTED - TYPE]" marker, returning the redacted string.
func applyRedactions(text string, matches []Match) string {
	if len(matches) == 0 {
		return text
	}

	// Sort by start position and remove any overlaps
	sorted := make([]Match, len(matches))
	copy(sorted, matches)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Start < sorted[j].Start })

	// Remove overlapping sub-matches while keeping the first/leftmost
	var clean []Match
	for _, m := range sorted {
		if len(clean) > 0 && m.Start < clean[len(clean)-1].End {
			continue
		}
		clean = append(clean, m)
	}

	var b strings.Builder
	b.Grow(len(text))

	pos := 0
	for _, m := range clean {
		if m.Start > pos {
			b.WriteString(text[pos:m.Start])
		}
		b.WriteString("[REDACTED - ")
		b.WriteString(m.Type)
		b.WriteString("]")
		pos = m.End
	}
	if pos < len(text) {
		b.WriteString(text[pos:])
	}

	return b.String()
}
