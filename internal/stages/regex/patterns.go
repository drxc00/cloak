package regex

import (
	"regexp"
	"strings"
)

type Pattern struct {
	Type string
	// Priority determines the order in which patterns are applied.
	// This ensures that more specific high priority patterns are applied before more general low priority patterns.
	Priority int
	Regex    *regexp.Regexp
	Validate func(string) bool
}

// ipv6 pattern is a bit more complicated than the others, so we build it up from pieces.
// It matches full IPv6 addresses, including compressed forms and IPv4-mapped addresses.
var ipv6Pattern string = strings.Join([]string{
	`(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}`,
	`(?:[0-9a-f]{1,4}:){1,7}:`,
	`(?:[0-9a-f]{1,4}:){1,6}:[0-9a-f]{1,4}`,
	`(?:[0-9a-f]{1,4}:){1,5}(?::[0-9a-f]{1,4}){1,2}`,
	`(?:[0-9a-f]{1,4}:){1,4}(?::[0-9a-f]{1,4}){1,3}`,
	`(?:[0-9a-f]{1,4}:){1,3}(?::[0-9a-f]{1,4}){1,4}`,
	`(?:[0-9a-f]{1,4}:){1,2}(?::[0-9a-f]{1,4}){1,5}`,
	`[0-9a-f]{1,4}:(?:(?::[0-9a-f]{1,4}){1,6})`,
	`:(?:(?::[0-9a-f]{1,4}){1,7}|:)`,
	`(?:[0-9a-f]{1,4}:){6}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}`,
	`::(?:ffff(?::0{1,4})?:)?(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}`,
}, "|")

var Patterns = []Pattern{
	{Type: "EMAIL", Priority: 10, Regex: regexp.MustCompile(`\b[\w.+-]+@[\w-]+\.[\w.-]+\b`)},
	{Type: "PHONE", Priority: 10, Regex: regexp.MustCompile(
		// +<cc> followed by 7-15 digits (no parens): +44 20 7946 0958
		`\+\d(?:[\s.\-]?\d){6,14}\b` +
			// +<cc> with parenthesised area code: +1 (415) 555-0198
			`|\+\d[\s.\-]?\(\d{3}\)[\s.\-]?\d{3}[\s.\-]\d{4}\b` +
			// Bare parenthesised (US/CA): (415) 555-0198
			`|\(\d{3}\)[\s.\-]?\d{3}[\s.\-]\d{4}\b`,
	), Validate: IsPhoneNumber},
	{Type: "IPv4", Priority: 20, Regex: regexp.MustCompile(`\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b`)},
	{Type: "IPv6", Priority: 20, Regex: regexp.MustCompile("(?i)(?:" + ipv6Pattern + ")")},
	{Type: "SSN", Priority: 20, Regex: regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`)},
	{Type: "MAC_ADDRESS", Priority: 20, Regex: regexp.MustCompile(`\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b`)},
	{Type: "IBAN", Priority: 20, Regex: regexp.MustCompile(`\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b`), Validate: IsValidIBAN},
	{Type: "CREDIT_CARD", Priority: 15, Regex: regexp.MustCompile(`\b\d(?:[ -]?\d){12,18}\b`), Validate: LuhnCheck},
	{Type: "JWT", Priority: 5, Regex: regexp.MustCompile(`\beyJ[\w-]+\.[\w-]+\.[\w-]+\b`)},
	{Type: "PRIVATE_KEY", Priority: 1, Regex: regexp.MustCompile(`(?s)-----BEGIN [A-Z ]+-----.*?-----END [A-Z ]+-----`)},
}
