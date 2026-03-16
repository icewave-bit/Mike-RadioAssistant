def main() -> None:
    # Print a simple welcome message, then start the CLI pipeline.
    from main import main as welcome_main
    from radiobuddy.cli import main as cli_main

    welcome_main()
    cli_main(["run", "--interactive-devices"])


if __name__ == "__main__":
    main()

