package regex

import "testing"

func TestLuhnCheck(t *testing.T) {
	tests := []struct {
		name   string
		number string
		want   bool
	}{
		// Valid credit card numbers
		{"valid visa", "4111111111111111", true},
		{"valid mastercard", "5555555555554444", true},
		{"valid amex", "378282246310005", true},
		{"valid discover", "6011111111111117", true},
		{"valid with spaces", "4111 1111 1111 1111", true},
		{"valid with dashes", "4111-1111-1111-1111", true},
		{"valid with mixed separators", "4111 1111-1111 1111", true},

		// Invalid numbers
		{"invalid one digit off", "4111111111111112", false},
		{"invalid all same", "1111111111111111", false},
		{"empty string", "", false},
		{"too short (<13 digits)", "4992739871", false},
		{"only non-digit chars", "----", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := LuhnCheck(tt.number)
			if got != tt.want {
				t.Errorf("LuhnCheck(%q) = %v, want %v", tt.number, got, tt.want)
			}
		})
	}
}
