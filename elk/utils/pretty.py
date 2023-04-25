# Kind of kickass that this file has no imports

# ANSI color codes for use in terminal output.
COLOR_CODES = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
}


def colorize(message: str, color: str) -> str:
    """Colorize a message for terminal output."""
    # Get the ANSI color code based on the human-readable color name.
    code = COLOR_CODES.get(color.lower())
    if code is None:
        raise ValueError(f"Invalid color name: {color}")

    # Construct and return the colored message.
    return f"\033[{code}m{message}\033[0m"
