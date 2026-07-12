package main

import (
	"context"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/drxc00/cloak/internal/pipeline"
	"github.com/drxc00/cloak/internal/stages/ner"
	"github.com/drxc00/cloak/internal/stages/regex"
	"github.com/drxc00/cloak/internal/stages/secrets"
)

var (
	dryRun   bool
	thorough bool
	disabled []string
	text     string
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
			pipeline.WithThorough(thorough),
		}
		if len(disabled) > 0 {
			opts = append(opts, pipeline.WithDisabled(disabled...))
		}
		config := pipeline.NewConfig(opts...)

		p := pipeline.New(
			config,
			regex.NewStage(),
			secrets.NewStage(),
			ner.NewStage(),
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
	redactCmd.Flags().BoolVar(&thorough, "thorough", false, "Enable LLM fallback (NER) stage")
	redactCmd.Flags().StringSliceVar(&disabled, "disable", nil, "Disable specific detector types")
	redactCmd.Flags().StringVarP(&text, "text", "t", "", "Redact inline text instead of a file")
	rootCmd.AddCommand(redactCmd)
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
