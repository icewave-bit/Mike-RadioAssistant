from __future__ import annotations

def main() -> None:
    """
    List available input/output audio devices so you can configure .env.
    """
    from .cli import main as cli_main

    cli_main(["list-devices"])


if __name__ == "__main__":
    main()

