package secrets

import (
	"context"
	"sort"
	"strings"

	"github.com/drxc00/cloak/internal/pipeline"
)

// Stage detects high-entropy strings (API keys, tokens, passwords) using
// vendor-prefix regexes, a semi-generic keyword+entropy pattern, and
// Shannon-entropy analysis. It runs after the regex stage so deterministic
// patterns take priority.
type Stage struct{}

func NewStage() *Stage { return &Stage{} }

// Detect processes text line by line. Secrets (API keys, tokens,
// passwords) never span newlines, so splitting eliminates cross-line
// matching edge cases and keeps key names on other lines visible.
func (s *Stage) Detect(_ context.Context, text string, claimed []pipeline.Match) []pipeline.Match {
	rules := Rules()
	if len(rules) == 0 {
		return nil
	}

	// Seed occupied spans with already-claimed matches from earlier stages
	// (global byte offsets).
	occupied := make([]span, 0, len(claimed))
	for _, m := range claimed {
		occupied = append(occupied, span{m.Start, m.End})
	}

	var matches []pipeline.Match
	offset := 0

	// SplitAfter preserves the \n delimiter so byte offsets stay accurate.
	lines := strings.SplitAfter(text, "\n")
	for _, line := range lines {
		if len(line) == 0 {
			continue
		}

		lower := strings.ToLower(line)

		// Track line-local occupied spans to prevent intra-line overlap.
		lineOccupied := make([]span, 0)

		for _, rule := range rules {
			if len(rule.Keywords) > 0 && !containsAny(lower, rule.Keywords) {
				continue
			}

			locs := rule.Regex.FindAllStringSubmatchIndex(line, -1)
			for _, loc := range locs {
				fullStart, fullEnd := loc[0], loc[1]

				// Within-line dedup.
				if overlapsAny(fullStart, fullEnd, lineOccupied) {
					continue
				}

				// Check global overlaps (from earlier stages or prior lines).
				globalStart, globalEnd := offset+fullStart, offset+fullEnd
				if overlapsAny(globalStart, globalEnd, occupied) {
					continue
				}

				secret, secStart, secEnd := extractSecret(line, loc, rule.SecretGroup)

				if rule.MinEntropy > 0 {
					if ent := shannonEntropy(secret); ent < rule.MinEntropy {
						continue
					}
				}

				if rule.Allowlist != nil && isAllowed(secret, rule.Allowlist) {
					continue
				}

				lineOccupied = append(lineOccupied, span{fullStart, fullEnd})
				occupied = append(occupied, span{globalStart, globalEnd})
				matches = append(matches, pipeline.Match{
					Type:  rule.Type,
					Start: offset + secStart,
					End:   offset + secEnd,
					Text:  secret,
				})
			}
		}

		offset += len(line)
	}

	sort.Slice(matches, func(i, j int) bool { return matches[i].Start < matches[j].Start })
	return matches
}

type span struct{ start, end int }

func overlapsAny(start, end int, spans []span) bool {
	for _, s := range spans {
		if start < s.end && s.start < end {
			return true
		}
	}
	return false
}

func containsAny(s string, keywords []string) bool {
	for _, kw := range keywords {
		if strings.Contains(s, kw) {
			return true
		}
	}
	return false
}

// extractSecret pulls the secret value and its byte span from a regex
// submatch index slice. SecretGroup 0 means "use the full match"; >0 means
// "use that capture group".
func extractSecret(text string, loc []int, group int) (string, int, int) {
	if group > 0 {
		idx := group * 2
		if idx+1 < len(loc) && loc[idx] >= 0 {
			return text[loc[idx]:loc[idx+1]], loc[idx], loc[idx+1]
		}
	}
	return text[loc[0]:loc[1]], loc[0], loc[1]
}

// isAllowed returns true when the secret should be suppressed by the
// allowlist (exact match, stopword substring, or allowlist regex match).
func isAllowed(secret string, a *Allowlist) bool {
	if a == nil {
		return false
	}

	if a.ExactSecrets != nil && a.ExactSecrets[secret] {
		return true
	}

	if a.StopWords != nil {
		lower := strings.ToLower(secret)
		for w := range a.StopWords {
			// Match as a whole word only
			// "example" but not "EXAMPLEKEY"
			if matchWord(lower, w) {
				return true
			}
		}
	}

	for _, re := range a.Regexes {
		if re.MatchString(secret) {
			return true
		}
	}

	return false
}

// matchWord returns true when w appears as a whole word in s. A word
// boundary is any non-letter character or start/end of string.
func matchWord(s, w string) bool {
	if len(w) == 0 {
		return false
	}
	for {
		i := strings.Index(s, w)
		if i == -1 {
			return false
		}
		// Check left boundary.
		if i > 0 && isLetter(s[i-1]) {
			s = s[i+1:]
			continue
		}
		// Check right boundary.
		end := i + len(w)
		if end < len(s) && isLetter(s[end]) {
			s = s[i+1:]
			continue
		}
		return true
	}
}

func isLetter(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= '0' && b <= '9') || b == '_'
}
