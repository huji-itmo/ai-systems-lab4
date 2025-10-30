import numpy as np
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import (
    Static,
    Input,
    Label,
)
from textual.validation import Number, Function

from tui.helper import sanitize_id


class RegressionTab(Static):
    def __init__(
        self,
        model_name: str,
        feature_names: list[str],
        coeffs: np.ndarray,
        r: float,
        r_squared: float,
        initial_values: dict[str, float] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.feature_names = feature_names
        self.coeffs = coeffs
        self.r = r
        self.r_squared = r_squared
        self.initial_values = initial_values or {}
        self.inputs = {}

    def compose(self) -> ComposeResult:
        yield Label(f"[b]{self.model_name}[/b]", classes="title")

        # Model fit statistics
        yield Label(
            f"Correlation (r): {self.r:.4f}", id=f"r-{sanitize_id(self.model_name)}"
        )
        yield Label(
            f"Coefficient of Determination (R²): {self.r_squared:.4f}",
            id=f"r2-{sanitize_id(self.model_name)}",
        )
        yield Label("")  # spacer

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
            "Predicted Performance Index: —",
            id=f"prediction-{sanitize_id(self.model_name)}",
        )
        yield self.prediction_label

    def on_mount(self) -> None:
        # Set initial values if provided
        for feat, widget in self.inputs.items():
            if feat in self.initial_values:
                val = self.initial_values[feat]
                # Format binary features as "0"/"1" strings
                if "Activities" in feat:
                    widget.value = str(int(val))
                else:
                    widget.value = str(val)
        # Trigger initial prediction
        self.predict()

    def predict(self):
        try:
            values = []
            for feat in self.feature_names:
                val = self.inputs[feat].value.strip()
                if not val:
                    raise ValueError("Missing input")
                values.append(float(val))
            x = np.array(values)
            pred = self.coeffs[0] + np.dot(self.coeffs[1:], x)
            self.prediction_label.update(f"🎯 Predicted Performance Index: {pred:.2f}")
        except Exception:
            self.prediction_label.update("Predicted Performance Index: —")

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed):
        self.predict()
