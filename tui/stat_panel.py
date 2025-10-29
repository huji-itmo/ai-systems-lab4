from textual.app import ComposeResult
from textual.widgets import (
    Static,
    Label,
)


class StatPanel(Static):
    DEFAULT_CSS = """
    StatPanel {
        width: 1fr;
        height: auto;
        padding: 1 2;
        margin: 1 0;
        background: $panel;
        border: round $primary;
        color: $foreground;
    }
    StatPanel > Label {
        width: 100%;
    }
    """

    def __init__(self, var_name: str, stats: dict, **kwargs):
        super().__init__(**kwargs)
        self.var_name = var_name
        self.stats = stats

    def compose(self) -> ComposeResult:
        yield Label(f"[b]📈 {self.var_name}[/b]")

        if "error" in self.stats:
            yield Label(f"❌ Error: {self.stats['error']}", classes="error")
            return

        s = self.stats
        lines = [
            f"Count (n):          {s['count']}",
            f"Mean:               {s['mean']:.4f}",
            f"Median:             {s['median']:.4f}",
            (
                f"Mode:               {s['mode']:.4f}"
                if s["mode"] is not None
                else "Mode:               —"
            ),
            f"Std:                {s['std']:.4f}",
            f"Std (corrected):    {s['std_corrected']:.4f}",
            f"Range:              {s['range']:.4f}",
            f"Min:                {s['min']:.4f}",
            f"Max:                {s['max']:.4f}",
        ]

        for line in lines:
            yield Label(line)
