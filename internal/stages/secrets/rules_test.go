package secrets

import (
	"strings"
	"testing"
)

// ruleCase pairs a rule Type with inputs the rule must (or must not) match.
type ruleCase struct {
	ruleType string
	tp       []string // true positives the rule MUST match these
	fp       []string // false positives the rule MUST NOT match these
}

func TestVendorRules_Regex(t *testing.T) {
	rules := Rules()
	ruleMap := make(map[string]Rule, len(rules))
	for _, r := range rules {
		ruleMap[r.Type] = r
	}

	cases := []ruleCase{
		{
			ruleType: "GITHUB_PAT",
			tp: []string{
				"ghp_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz",
				"ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
				"token=ghp_000000000000000000000000000000000000",
			},
			fp: []string{
				"ghp_short",
				"ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",   // exact placeholder
				"not a token ghp but not enough chars after", // no underscore
				"GHP_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz",   // uppercase prefix
			},
		},
		{
			ruleType: "GITHUB_FINE_GRAINED_PAT",
			tp: []string{
				"github_pat_11AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA0",
			},
			fp: []string{
				"github_pat_too_short",
				"github_pat_",
			},
		},
		{
			ruleType: "GITHUB_OAUTH",
			tp: []string{
				"gho_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz",
			},
			fp: []string{
				"gho_short",
				"gho_12345678901234567890123456789012345", // 35 chars after gho_, not 36
			},
		},
		{
			ruleType: "GITHUB_APP_TOKEN",
			tp: []string{
				"ghu_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz",
				"ghs_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz",
			},
			fp: []string{
				"ghu_short",
				"ghx_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz", // wrong prefix
			},
		},
		{
			ruleType: "GITHUB_REFRESH_TOKEN",
			tp: []string{
				"ghr_1A2b3C4d5E6f7G8h9I0jklMNOPQRSTUVwxyz",
			},
			fp: []string{
				"ghr_short",
			},
		},
		{
			ruleType: "GITLAB_TOKEN",
			tp: []string{
				"glpat-abcdefghijklmnopqrstuvwxyz1234567890",
				"gldt-abcdefghijklmnopqrstuvwxyz1234567890",
				"glrt-abcdefghijklmnopqrstuvwxyz1234567890",
				"glagent-abcdefghijklmnopqrstuvwxyz1234567890",
			},
			fp: []string{
				"glxyz-notatoken",
				"glpa-abcdefghij", // missing 't'
			},
		},
		{
			ruleType: "AWS_ACCESS_KEY",
			tp: []string{
				"AKIALALEMEL33243OLIB",
				"ASIAZZZZZZZZZZZZZZZZ",
				"ABIAAAAAAAAAAAAAAAAA",
				"ACCAAAAAAAAAAAAAAAAA",
				"A3TAAAAAAAAAAAAAAAAA",
			},
			fp: []string{
				"AKIA0000000000000000", // 0 not in AWS Base32 alphabet
				"AKIA1111111111111111", // 1 not valid
			},
		},
		{
			ruleType: "OPENAI_API_KEY",
			tp: []string{
				"sk-1234567890abcdefghijT3BlbkFJ1234567890abcdefghij",
				"sk-proj-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaT3BlbkFJbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
				"sk-svcacct-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaT3BlbkFJbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
			},
			fp: []string{
				"sk-1234567890abcdefghij1234567890abcdefghij", // no T3BlbkFJ marker
				"not-a-key",
			},
		},
		{
			ruleType: "SLACK_BOT_TOKEN",
			tp: []string{
				"xoxb-781236542736-2364535789652-GkwFDQoHqzXDVsC6GzqYUypD",
				"xoxb-1234567890-1234567890123-abc123",
			},
			fp: []string{
				"xoxb-short",
				"xoxb-123-456-abc", // numbers too short
			},
		},
		{
			ruleType: "SLACK_USER_TOKEN",
			tp: []string{
				"xoxp-41684372915-1320496754-45609968301-e708ba56e1517a99f6b5fb07349476ef",
				"xoxe-12345678901-1234567890-12345678901-abcdefghijklmnopqrstuvwxyz0123",
			},
			fp: []string{
				"xoxp-short-token",
			},
		},
		{
			ruleType: "SLACK_WEBHOOK_URL",
			tp: []string{
				"hooks.slack.com/services/T024TTTTT/BBB72BBL/AZAAA9u0pA4ad666eMgbi555",
				"https://hooks.slack.com/workflows/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",
			},
			fp: []string{
				"hooks.slack.com/not-a-real-path",
				"slack.com/oauth/authorize", // not hooks subdomain
			},
		},
		{
			ruleType: "STRIPE_ACCESS_TOKEN",
			tp: []string{
				"sk_test_51Hc8n29fJd8sKdlA0293KfnaLc12345",
				"sk_live_abcdefghijklmnopqrstuvwxyz",
				"rk_prod_abcdefghijklmnopqrstuvwxyz0123456789",
			},
			fp: []string{
				"sk_prod_short",            // only 5 chars after prefix
				"pk_test_abcdefghijklmnop", // publishable key, not secret
			},
		},
		{
			ruleType: "GENERIC_API_KEY",
			tp: []string{
				`api_key = "abcdefghijklmnopqrstuvwxyz0123456789"`,
				`token: abcdefghijklmnopqrstuvwxyz0123456789`,
				`password := "SuperSecret2345678"`,
			},
			fp: []string{
				`api_version = "1.0.0"`, // too short (<10 chars secret)
				`token = "example"`,     // stopword
				`key = "onlyletters"`,   // alphabetic-only regex suppresses
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.ruleType, func(t *testing.T) {
			rule, ok := ruleMap[tc.ruleType]
			if !ok {
				t.Fatalf("rule %q not found in Rules()", tc.ruleType)
			}

			for _, input := range tc.tp {
				if !rule.Regex.MatchString(input) {
					t.Errorf("TRUE POSITIVE missed: %q did not match regex for %s", input, tc.ruleType)
				}
			}

			for _, input := range tc.fp {
				if rule.Regex.MatchString(input) {
					// For rules with allowlists, the regex matching is expected;
					// the allowlist suppresses it later. Skip if it's an allowlist case.
					if rule.Allowlist != nil {
						// Extract the secret and check if allowlist would suppress it.
						loc := rule.Regex.FindStringSubmatchIndex(input)
						if loc != nil {
							secret, _, _ := extractSecret(input, loc, rule.SecretGroup)
							if isAllowed(secret, rule.Allowlist) {
								continue // allowlist correctly suppresses
							}
						}
					}
					t.Errorf("FALSE POSITIVE not caught: %q matched regex for %s", input, tc.ruleType)
				}
			}
		})
	}
}

func TestVendorRules_Keywords(t *testing.T) {
	// Every rule with keywords should have at least one keyword
	// that appears in the rule's own regex source (case-insensitive).
	// This is a sanity check, not a requirement — keywords may match
	// surrounding context that the regex doesn't embed literally.
	rules := Rules()
	for _, r := range rules {
		if len(r.Keywords) == 0 {
			continue // generic rule, fine
		}
		pattern := strings.ToLower(r.Regex.String())
		found := false
		for _, kw := range r.Keywords {
			k := strings.ToLower(kw)
			// Keyword may appear in the regex literally or in escaped form.
			// Also check common regex escaping: \. for ., \- for -, etc.
			if strings.Contains(pattern, k) ||
				strings.Contains(pattern, strings.ReplaceAll(k, ".", `\.`)) ||
				strings.Contains(pattern, strings.ReplaceAll(k, "-", `\-`)) {
				found = true
				break
			}
		}
		if !found {
			// Keywords can legitimately match context outside the regex
			// (e.g., "twilio" near a Twilio SID). These are fine.
			t.Logf("rule %s: keywords %v not found literally in regex — may match context", r.Type, r.Keywords)
		}
	}
}

func TestVendorRules_EntropyThreshold(t *testing.T) {
	// Every rule should have a reasonable MinEntropy (0 = disabled, or > 0 and ≤ 5.0).
	rules := Rules()
	for _, r := range rules {
		if r.MinEntropy > 5.0 {
			t.Errorf("rule %s: MinEntropy is %v, suspiciously high", r.Type, r.MinEntropy)
		}
	}
}

func TestRules_NoDuplicateTypes(t *testing.T) {
	rules := Rules()
	seen := make(map[string]bool, len(rules))
	for _, r := range rules {
		if seen[r.Type] {
			t.Errorf("duplicate rule Type: %s", r.Type)
		}
		seen[r.Type] = true
	}
}
