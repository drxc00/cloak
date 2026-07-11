package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
	Use:   "cloak [input]",
	Short: "Cloak is a tool for obfuscating sensitive information.",
	Long: `
	Cloak is a lightweight command-line tool that filters sensitive information out of your data before it ever reaches an AI model. 
	Logs, tickets, config files, chat exports, code — anything you'd normally think twice about pasting into a prompt.
	Cloak scans it first and strips out names, emails, IP addresses, API keys, passwords, credentials, and dozens of other sensitive patterns, replacing each with a clear [REDACTED - TYPE] marker
	`,
	Args: cobra.ArbitraryArgs,
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}
