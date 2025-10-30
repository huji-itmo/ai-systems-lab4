# tui/kNN_tab.py
from typing import List
import numpy as np
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static, Input, Label
from textual.validation import Number

from tui.helper import sanitize_id
from tui.kNN import predict_knn


class KNNTab(Static):
    def __init__(
        self,
        model_name: str,
        feature_names: list[str],
        x_data: List[List[float]],
        y_data: List[float],
        initial_values: dict[str, float] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.feature_names = feature_names
        self.x_data = np.array(x_data, dtype=float)
        self.y_data = np.array(y_data, dtype=float)
        self.initial_values = initial_values or {}
        self.inputs = {}
        self.k_values = [3, 5, 10]
        self.prediction_labels = {}

    def compose(self) -> ComposeResult:
        yield Label(f"[b]{self.model_name}[/b]", classes="title")
        yield Label("")  # spacer

        for feat in self.feature_names:
            label = Label(f"{feat}:")
            # All diabetes features are non-negative numbers
            validator = Number(minimum=0)
            safe_id = f"input-{sanitize_id(feat)}"
            input_widget = Input(
                placeholder=f"Enter {feat}",
                validators=[validator],
                id=safe_id,
            )
            self.inputs[feat] = input_widget
            yield Horizontal(label, input_widget)

        yield Label("")  # spacer

        for k in self.k_values:
            label_widget = Label(
                "Predicted Diabetes: —",
                id=f"prediction-k{k}-{sanitize_id(self.model_name)}",
            )
            self.prediction_labels[k] = label_widget
            yield label_widget

    def on_mount(self) -> None:
        for feat, widget in self.inputs.items():
            if feat in self.initial_values:
                widget.value = str(self.initial_values[feat])
        self.predict()

    def predict(self):
        try:
            values = []
            for feat in self.feature_names:
                val = self.inputs[feat].value.strip()
                if not val:
                    raise ValueError("Missing input")
                values.append(float(val))
            x_input = np.array(values)

            predictions = predict_knn(
                x_input=x_input,
                x_data=self.x_data,
                y_data=self.y_data,
                k_values=self.k_values,
            )

            for k, text in predictions.items():
                self.prediction_labels[k].update(text)

        except Exception:
            for k in self.k_values:
                self.prediction_labels[k].update("Predicted Diabetes: —")

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed):
        self.predict()
