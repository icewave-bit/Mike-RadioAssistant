def main() -> None:
    from radiobuddy.cli import main as cli_main

    cli_main(["run", "--interactive-devices"])


if __name__ == "__main__":
    main()

