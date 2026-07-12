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
