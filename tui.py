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
from textual.binding import Binding

from tui.dynamic_kNN_tab import DynamickNNTab
from tui.regression import multiple_linear_regression_scalar
from tui.analyze_dataset import analyze_dataset
from tui.kNN_tab import KNNTab
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
        Binding("q", "quit", "Quit"),
        Binding("escape", "unfocus", "Unfocus", show=False),  # <-- Add this
    ]

    def __init__(self, csv_path: str = "Student_Performance.csv"):
        super().__init__()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        self.all_features = [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "Pedigree",
            "Age",
        ]
        self.target = "Outcome"

        # Sample one random row for initial values
        random_row = self.df.sample(n=1).iloc[0]
        self.model1_initial = {feat: random_row[feat] for feat in self.all_features}

        # Model 1: Full
        self.X_all = self.df[self.all_features].to_numpy()
        self.y = self.df[self.target].to_numpy()

        # Analyze dataset
        self.analysis_results = analyze_dataset(csv_path)

    def on_mount(self) -> None:
        self.register_theme(get_theme())
        self.theme = "monokai"

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("📊 Dataset Statistics"):
                with VerticalScroll():
                    for var_name, stats in self.analysis_results.items():
                        yield StatPanel(var_name=var_name, stats=stats)

            with TabPane("🔍 kNN: All Features (8)"):
                with VerticalScroll():
                    yield KNNTab(
                        model_name="kNN – All 8 Features",
                        feature_names=self.all_features,
                        x_data=self.X_all.tolist(),
                        y_data=self.y.tolist(),
                        initial_values=self.model1_initial,
                    )

            with TabPane("⚙️ kNN: Custom Feature Selection"):
                with VerticalScroll():
                    yield DynamickNNTab(
                        model_name="kNN – Select Features",
                        all_feature_names=[
                            "Pregnancies",
                            "Glucose",
                            "BloodPressure",
                            "SkinThickness",
                            "Insulin",
                            "BMI",
                            "Pedigree",
                            "Age",
                        ],
                        x_data=self.X_all.tolist(),  # Full x_data
                        y_data=self.y.tolist(),
                        initial_values=self.model1_initial,  # Same initial row
                    )

        yield Footer()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()


if __name__ == "__main__":
    MyApp("diabetes.csv").run()
