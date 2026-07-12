package secrets

import "regexp"

// setOf returns a lookup set from the given values.
func setOf(vals ...string) map[string]bool {
	m := make(map[string]bool, len(vals))
	for _, v := range vals {
		m[v] = true
	}
	return m
}

// genericAllowlist returns the allowlist used by the generic-api-key rule.
// It rejects purely-alphabetic tokens, known placeholder secrets, and a
// curated set of common English / tech words that frequently appear next to
// assignment operators but are not credentials.
func genericAllowlist() *Allowlist {
	return &Allowlist{
		Regexes: []*regexp.Regexp{
			regexp.MustCompile(`^[a-zA-Z_.-]+$`),
		},
		StopWords:    defaultStopWords,
		ExactSecrets: setOf("AKIAIOSFODNN7EXAMPLE", "12345678901234567890123456789012", "00000000000000000000000000000000"),
	}
}

var defaultStopWords = setOf(
	"about", "abstract", "academy", "account", "action", "active",
	"activity", "admin", "adobe", "advanced", "adventure", "agent",
	"amazon", "android", "angular", "animation", "answer", "apache",
	"apple", "archive", "article", "asset", "author", "auto",
	"awesome", "azure",
	"base", "basic", "beta", "better", "binary", "block", "blog",
	"board", "book", "branch", "browser", "build", "builder", "bundle",
	"cache", "calendar", "center", "channel", "chart", "chat", "check",
	"chrome", "classic", "clean", "client", "clone", "cloud", "cluster",
	"code", "color", "command", "comment", "commit", "common",
	"community", "component", "computer", "config", "connect",
	"container", "content", "control",
	"data", "database", "date", "debug", "default", "demo", "deploy",
	"design", "desktop", "develop", "device", "directory", "display",
	"docker", "document", "domain", "download", "driver", "dynamic",
	"email", "engine", "error", "event", "example", "exchange",
	"export", "extension", "external",
	"facebook", "factory", "feature", "field", "file", "filter",
	"firefox", "folder", "format", "framework", "free", "function",
	"future",
	"gallery", "game", "general", "generator", "generic",
	"github", "gitlab", "google", "graphic", "group",
	"hello", "helper", "history", "home", "homepage",
	"image", "import", "index", "info", "input", "install",
	"interface", "internal",
	"language", "latest", "layout", "level",
	"license", "light", "link", "linux", "list", "local", "login",
	"mail", "manager", "manual", "master", "media",
	"memory", "message", "method", "middleware", "mirror",
	"mobile", "model", "modern", "module", "monitor", "movie", "music",
	"name", "native", "network", "next", "node", "number",
	"object", "online", "open", "option", "order", "original",
	"output",
	"package", "page", "paper", "parser", "password", "path",
	"pattern", "payment", "people", "phone", "photo", "pipeline",
	"platform", "player", "plugin", "portal", "power", "preview",
	"private", "product", "profile", "program", "project", "protocol",
	"public",
	"query", "queue",
	"random", "readme", "registry", "release", "remote", "report",
	"request", "resource", "response", "result",
	"sample", "schema", "script", "search", "secret", "security",
	"select", "server", "service", "session", "setting", "setup",
	"shell", "shop", "simple", "single", "social", "software",
	"solution", "source", "spring", "stack", "standard", "start",
	"static", "storage", "store", "string", "structure", "studio",
	"style", "summary", "support", "system",
	"table", "target", "task", "team", "template", "terminal",
	"testing", "theme", "title", "token", "tool", "tracker",
	"training", "tutorial", "twitter",
	"update", "upload", "user", "utility",
	"value", "version", "video", "view", "virtual", "visual",
	"warning", "weather", "web", "website", "welcome",
	"widget", "window", "workflow", "workshop", "world",
	"xxxxxx", "000000",
)
