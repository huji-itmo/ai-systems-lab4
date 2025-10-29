import numpy as np
import pandas as pd
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    TabbedContent,
    TabPane,
    Static,
    Input,
    Label,
    Footer,
    Header,
)
from textual.validation import Number, Function

from analyze_dataset import analyze_dataset
from regression import multiple_linear_regression_scalar

from textual.theme import Theme

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
        "footer-key-foreground": "#cba6f7",  # Mauve
        "input-selection-background": "#89b4fa 35%",  # Blue (35% opacity)
    },
)


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map(
        {"Yes": 1, "No": 0}
    )
    return df


def sanitize_id(text: str) -> str:
    """Convert a string into a valid Textual widget ID."""
    # Replace spaces and invalid chars with underscores
    sanitized = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in text)
    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


class RegressionTab(Static):
    """A reusable tab for regression input and live prediction."""

    def __init__(
        self,
        model_name: str,
        feature_names: list[str],
        coeffs: np.ndarray,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.feature_names = feature_names
        self.coeffs = coeffs  # [intercept, beta1, beta2, ...]
        self.inputs = {}

    def compose(self) -> ComposeResult:
        yield Label(f"[b]{self.model_name}[/b]", classes="title")
        for feat in self.feature_names:
            label = Label(f"{feat}:")
            validator = (
                Number(minimum=0)
                if "Activities" not in feat
                else Function(lambda x: x in ("0", "1"), "Enter 0 or 1")
            )
            safe_id = f"input-{sanitize_id(feat)}"
            input_widget = Input(
                placeholder=f"Enter {feat}", validators=[validator], id=safe_id
            )
            self.inputs[feat] = input_widget
            yield Horizontal(label, input_widget)

        self.prediction_label = Label(
            "Prediction: —", id=f"prediction-{sanitize_id(self.model_name)}"
        )
        yield self.prediction_label

    def predict(self):
        try:
            values = []
            for feat in self.feature_names:
                val = self.inputs[feat].value.strip()
                if not val:
                    raise ValueError("Missing input")
                if feat == "Extracurricular Activities":
                    values.append(float(val))
                else:
                    values.append(float(val))
            x = np.array(values)
            pred = self.coeffs[0] + np.dot(self.coeffs[1:], x)
            self.prediction_label.update(f"🎯 Predicted Performance Index: {pred:.2f}")
        except Exception:
            self.prediction_label.update("Prediction: —")

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed):
        self.predict()


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
    """

    from textual.app import App

    class MyApp(App):
        def on_mount(self) -> None:
            # Register the theme
            self.register_theme(catppuccin_mocha)

            # Set the app's theme
            self.theme = "catppuccin-mocha"

    def __init__(self, csv_path: str = "Student_Performance.csv"):
        super().__init__()
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df = preprocess_features(self.df)

        # Full feature set
        self.all_features = [
            "Hours Studied",
            "Previous Scores",
            "Extracurricular Activities",
            "Sleep Hours",
            "Sample Question Papers Practiced",
        ]
        self.target = "Performance Index"

        # Prepare data
        X_full = self.df[self.all_features].values
        y = self.df[self.target].values

        # Model 1: Full
        coeffs1 = multiple_linear_regression_scalar(y, X_full)
        self.model1_features = self.all_features
        self.model1_coeffs = coeffs1

        # Model 2: 3 features
        feats2 = ["Hours Studied", "Previous Scores", "Sleep Hours"]
        X2 = self.df[feats2].values
        coeffs2 = multiple_linear_regression_scalar(y, X2)
        self.model2_features = feats2
        self.model2_coeffs = coeffs2

        # Model 3: 4 features
        feats3 = [
            "Previous Scores",
            "Extracurricular Activities",
            "Sample Question Papers Practiced",
            "Sleep Hours",
        ]
        X3 = self.df[feats3].values
        coeffs3 = multiple_linear_regression_scalar(y, X3)
        self.model3_features = feats3
        self.model3_coeffs = coeffs3

        # Analyze dataset output (as string)
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        try:
            analyze_dataset(csv_path)
        finally:
            sys.stdout = old_stdout
        self.analysis_output = captured_output.getvalue()

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("📊 Dataset Analysis"):
                yield VerticalScroll(Static(self.analysis_output, markup=False))

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


if __name__ == "__main__":
    MyApp().run()
