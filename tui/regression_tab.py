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
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.feature_names = feature_names
        self.coeffs = coeffs
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
            "Predicted Performance Index: —",
            id=f"prediction-{sanitize_id(self.model_name)}",
        )
        yield self.prediction_label

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
            self.prediction_label.update("Prediction Performance Index: —")

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed):
        self.predict()
