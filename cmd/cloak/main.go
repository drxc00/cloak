package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"

	"github.com/spf13/cobra"

	"github.com/drxc00/cloak/internal/pipeline"
	"github.com/drxc00/cloak/internal/stages/ner"
	"github.com/drxc00/cloak/internal/stages/regex"
	"github.com/drxc00/cloak/internal/stages/secrets"
)

// Base URL for downloading sidecar binary artifacts. Override with CLOAK_RELEASE_URL.
const defaultReleaseURL = "https://github.com/drxc00/cloak/releases/latest/download"

// Base Hugging Face repo for downloading model artifacts.
// Override with CLOAK_HF_REPO (format: "<org>/<repo>").
const defaultHFRepo = "drxc0/cloak-ner-v1"

var (
	dryRun       bool
	thorough     bool
	disabled     []string
	text         string
	modelVariant string
)

var rootCmd = &cobra.Command{
	Use:   "cloak",
	Short: "Cloak is a tool for redacting sensitive information.",
	Long: `Cloak scans text, files, or stdin for sensitive data (names, emails,
IPs, API keys, credentials, etc.) and replaces each match with a
[REDACTED - TYPE] marker before the data ever reaches an AI model.`,
}

var redactCmd = &cobra.Command{
	Use:   "redact [flags] [input]",
	Short: "Redact sensitive information from text, files, or stdin.",
	Long: `Redact scans input for sensitive data (emails, IPs, SSNs, credit
cards, API keys, names, etc.) and replaces each match with a
[REDACTED - TYPE] marker.

Use --text for inline strings, pass a file path as argument, or pipe
via stdin.`,
	Args: cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		opts := []pipeline.Option{
			pipeline.WithDryRun(dryRun),
		}
		if len(disabled) > 0 {
			opts = append(opts, pipeline.WithDisabled(disabled...))
		}

		// NER is opt-in via --thorough and requires cloak-nerd to be installed.
		stages := []pipeline.Stage{regex.NewStage(), secrets.NewStage()}
		var nerStage *ner.Stage
		if thorough && ner.Installed() {
			opts = append(opts, pipeline.WithThorough(true))
			nerStage = ner.NewStage()
			stages = append(stages, nerStage)
		}
		defer nerStage.Close()

		config := pipeline.NewConfig(opts...)

		p := pipeline.New(
			config,
			stages...,
		)

		var input *pipeline.Input
		var err error
		switch {
		case text != "":
			input = pipeline.FromString(text)
		case len(args) == 1:
			input, err = pipeline.FromFile(args[0])
			if err != nil {
				return fmt.Errorf("open %q: %w", args[0], err)
			}
		default:
			input = pipeline.FromStdin()
		}

		result, err := p.Run(context.Background(), input)
		if err != nil {
			return err
		}

		fmt.Print(result.Redacted)
		return nil
	},
}

func init() {
	redactCmd.Flags().BoolVar(&dryRun, "dry-run", false, "Detect but do not redact")
	redactCmd.Flags().BoolVar(&thorough, "thorough", false, "Enable NER stage (names, addresses, usernames)")
	redactCmd.Flags().StringSliceVar(&disabled, "disable", nil, "Disable specific detector types")
	redactCmd.Flags().StringVarP(&text, "text", "t", "", "Redact inline text instead of a file")

	initCmd.Flags().StringVar(&modelVariant, "model", "edge", "Model variant to install: edge or full")

	rootCmd.AddCommand(redactCmd)
	rootCmd.AddCommand(initCmd)
}

// ---------------------------------------------------------------------------
// cloak init — download NER model + sidecar binary
// ---------------------------------------------------------------------------

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Download the NER model and inference engine",
	Long: `Downloads the PII token-classification model (ONNX) and the cloak-nerd
inference engine into ~/.cache/cloak/. After init, the --thorough flag on
'cloak redact' enables named-entity detection (names, addresses, usernames).

Requires an internet connection.

Two model variants are available:
  --model edge  (default)  Small distilbert, INT8 quantized — ~50 MB download.
  --model full             Larger deberta-v3, full precision — more accurate,
                           bigger download (~450 MB).

Run again to re-download and replace existing files.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		cacheDir, err := userCacheDir()
		if err != nil {
			return err
		}

		modelDir := filepath.Join(cacheDir, "models")
		binDir := filepath.Join(cacheDir, "bin")

		for _, d := range []string{modelDir, binDir} {
			if err := os.MkdirAll(d, 0755); err != nil {
				return fmt.Errorf("create %s: %w", d, err)
			}
		}

		// Validate --model.
		switch modelVariant {
		case "edge", "full":
		default:
			return fmt.Errorf("unknown model variant %q — must be 'edge' or 'full'", modelVariant)
		}

		// Model files come from Hugging Face Hub.
		hfRepo := os.Getenv("CLOAK_HF_REPO")
		if hfRepo == "" {
			hfRepo = defaultHFRepo
		}

		modelFiles := []string{"model.onnx", "tokenizer.json", "model_config.json"}
		for _, fn := range modelFiles {
			url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s/%s", hfRepo, modelVariant, fn)
			dest := filepath.Join(modelDir, fn)

			fmt.Printf("⟳  Downloading %s/%s …\n", modelVariant, fn)
			if err := downloadFile(url, dest); err != nil {
				return fmt.Errorf("download %s/%s: %w", modelVariant, fn, err)
			}
			fmt.Printf("   ✓  %s\n", fn)
		}

		// Sidecar binary still comes from GitHub Releases.
		releaseURL := os.Getenv("CLOAK_RELEASE_URL")
		if releaseURL == "" {
			releaseURL = defaultReleaseURL
		}

		nerdName := fmt.Sprintf("cloak-nerd-%s-%s", runtime.GOOS, runtime.GOARCH)
		nerdURL := fmt.Sprintf("%s/%s", releaseURL, nerdName)
		nerdDest := filepath.Join(binDir, nerdName)
		nerdCanon := filepath.Join(binDir, "cloak-nerd")

		fmt.Printf("⟳  Downloading %s …\n", nerdName)
		if err := downloadFile(nerdURL, nerdDest); err != nil {
			return fmt.Errorf("download %s: %w", nerdName, err)
		}

		os.Remove(nerdCanon)
		if err := os.Rename(nerdDest, nerdCanon); err != nil {
			return fmt.Errorf("rename: %w", err)
		}
		if err := os.Chmod(nerdCanon, 0755); err != nil {
			return fmt.Errorf("chmod: %w", err)
		}
		fmt.Printf("   ✓  cloak-nerd\n")

		// Write variant marker.
		if err := os.WriteFile(filepath.Join(modelDir, "variant.txt"), []byte(modelVariant), 0644); err != nil {
			return fmt.Errorf("write variant marker: %w", err)
		}

		fmt.Println()
		fmt.Println("NER stage is ready. Use --thorough with 'cloak redact' to enable.")
		fmt.Printf("Files installed in %s\n", cacheDir)
		return nil
	},
}

func userCacheDir() (string, error) {
	dir, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("user cache dir: %w", err)
	}
	return filepath.Join(dir, "cloak"), nil
}

func downloadFile(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s", resp.Status)
	}

	f, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("create: %w", err)
	}
	defer f.Close()

	if _, err := io.Copy(f, resp.Body); err != nil {
		return fmt.Errorf("write: %w", err)
	}

	return nil
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
