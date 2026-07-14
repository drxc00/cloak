package regex

import (
	"strings"

	"github.com/nyaruka/phonenumbers"
)

// LuhnCheck implements the Luhn algorithm to validate credit card numbers.
func LuhnCheck(number string) bool {
	var sum int
	var idx int

	// Process digits from right to left, skipping non-digit characters.
	for i := len(number) - 1; i >= 0; i-- {
		c := number[i]
		if c < '0' || c > '9' {
			continue
		}

		digit := int(c - '0')

		// Double every second digit (counting from the right).
		if idx%2 == 1 {
			digit *= 2
			if digit > 9 {
				digit -= 9
			}
		}

		sum += digit
		idx++
	}

	return sum%10 == 0 && idx >= 13
}

// IsValidIBAN checks the mod-97 checksum (ISO 7064 MOD 97-10) shared by all
// IBANs, so we don't redact every bare "two letters + digits" token as an
// account number.
func IsValidIBAN(s string) bool {
	if len(s) < 15 || len(s) > 34 {
		return false
	}

	rearranged := s[4:] + s[:4]

	feed := func(remainder, digit int) int { return (remainder*10 + digit) % 97 }

	remainder := 0
	for _, c := range rearranged {
		switch {
		case c >= 'A' && c <= 'Z':
			v := int(c-'A') + 10 // two-digit value: tens then units
			remainder = feed(remainder, v/10)
			remainder = feed(remainder, v%10)
		case c >= '0' && c <= '9':
			remainder = feed(remainder, int(c-'0'))
		default:
			return false
		}
	}

	return remainder == 1
}

// IsPhoneNumber validates a candidate phone number string using Google's
// libphonenumber. The candidate comes from one of two regexes:
//   - +<country code> <number> (auto-detects region from prefix)
//   - (XXX) XXX-XXXX (treated as US/CA)
func IsPhoneNumber(s string) bool {
	if strings.HasPrefix(s, "+") {
		num, err := phonenumbers.Parse(s, "")
		if err != nil {
			return false
		}
		return phonenumbers.IsPossibleNumber(num)
	}
	// Parenthesised format — treat as US/CA.
	num, err := phonenumbers.Parse(s, "US")
	if err != nil {
		return false
	}
	return phonenumbers.IsPossibleNumber(num)
}
