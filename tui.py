import numpy as np
import pandas as pd
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import (
    TabbedContent,
    TabPane,
    Footer,
    Header,
)
from textual import on

from regression import multiple_linear_regression_scalar
from tui.analyze_dataset import analyze_dataset
from tui.helper import preprocess_features
from tui.regression_tab import RegressionTab
from tui.stat_panel import StatPanel
from tui.theme import get_theme


class MyApp(App):
    CSS = """
    TabbedContent {
        height: 100%;
    }
    .title {
        margin-top: 1;
        text-style: bold;
    }
    Horizontal {
        height: 3;
        align: left middle;
    }
    Input {
        width: 30;
        margin-left: 2;
    }
    #prediction {
        margin-top: 1;
        padding: 1 2;
        background: $surface;
        border: tall $primary;
    }
    .error {
        color: $error;
    }
    """

    # Optional: Define key hints for the footer
    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self, csv_path: str = "Student_Performance.csv"):
        super().__init__()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df = preprocess_features(self.df)

        self.all_features = [
            "Hours Studied",
            "Previous Scores",
            "Extracurricular Activities",
            "Sleep Hours",
            "Sample Question Papers Practiced",
        ]
        self.target = "Performance Index"

        X_full = self.df[self.all_features].values
        y = np.array(self.df[self.target].values)

        # Model 1: Full
        self.model1_coeffs = multiple_linear_regression_scalar(y, X_full)
        self.model1_features = self.all_features

        # Model 2: 3 features
        feats2 = ["Hours Studied", "Previous Scores", "Sleep Hours"]
        X2 = self.df[feats2].values
        self.model2_coeffs = multiple_linear_regression_scalar(y, X2)
        self.model2_features = feats2

        # Model 3: 4 features
        feats3 = [
            "Previous Scores",
            "Extracurricular Activities",
            "Sample Question Papers Practiced",
            "Sleep Hours",
        ]
        X3 = self.df[feats3].values
        self.model3_coeffs = multiple_linear_regression_scalar(y, X3)
        self.model3_features = feats3

        # Analyze dataset
        self.analysis_results = analyze_dataset(csv_path)

    def on_mount(self) -> None:
        self.register_theme(get_theme())
        self.theme = "catppuccin-mocha"

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("📊 Dataset Analysis"):
                with VerticalScroll(id="analysis-container"):
                    for var_name, stats in self.analysis_results.items():
                        yield StatPanel(var_name, stats)

            with TabPane("📈 Regression 1 (Full)"):
                yield RegressionTab(
                    "Model 1: Full Features", self.model1_features, self.model1_coeffs
                )

            with TabPane("📈 Regression 2 (3 Features)"):
                yield RegressionTab(
                    "Model 2: Hours, Scores, Sleep",
                    self.model2_features,
                    self.model2_coeffs,
                )

            with TabPane("📈 Regression 3 (4 Features)"):
                yield RegressionTab(
                    "Model 3: Scores, Extra, Papers, Sleep",
                    self.model3_features,
                    self.model3_coeffs,
                )

        yield Footer()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


if __name__ == "__main__":
    MyApp().run()
