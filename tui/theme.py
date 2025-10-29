from textual.theme import Theme


def get_theme():
    catppuccin_mocha = Theme(
        name="catppuccin-mocha",
        primary="#cba6f7",  # Mauve
        secondary="#89b4fa",  # Blue
        accent="#f5c2e7",  # Pink
        foreground="#cdd6f4",  # Text
        background="#1e1e2e",  # Base
        success="#a6e3a1",  # Green
        warning="#f9e2af",  # Yellow
        error="#f38ba8",  # Red
        surface="#313244",  # Surface0
        panel="#45475a",  # Surface1
        dark=True,
        variables={
            "block-cursor-text-style": "none",
            "footer-key-foreground": "#cba6f7",
            "input-selection-background": "#89b4fa 35%",
        },
    )
    return catppuccin_mocha
