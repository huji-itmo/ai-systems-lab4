# tui/dynamic_knn_tab.py
from typing import List
import numpy as np
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, Input, Label, Checkbox
from textual.validation import Number

from tui.helper import sanitize_id
from tui.kNN import predict_knn


class DynamickNNTab(Static):
    def __init__(
        self,
        model_name: str,
        all_feature_names: List[str],
        x_data: List[List[float]],
        y_data: List[float],
        initial_values: dict[str, float] | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.all_feature_names = all_feature_names
        self.x_data_full = np.array(x_data, dtype=float)
        self.y_data = np.array(y_data, dtype=float)
        self.initial_values = initial_values or {}
        self.k_values = [3, 5, 10]
        self.selected_features = set(all_feature_names)  # Start with all selected
        self.inputs = {}
        self.checkboxes = {}
        self.prediction_labels = {}

    def compose(self) -> ComposeResult:
        yield Label(f"[b]{self.model_name}[/b]", classes="title")
        yield Label("✅ Select features to use for prediction:", classes="subtitle")

        # Yield checkboxes directly inside a Vertical
        with Vertical(id="feature-checkboxes"):
            for feat in self.all_feature_names:
                cb = Checkbox(feat, value=True, id=f"cb-{sanitize_id(feat)}")
                self.checkboxes[feat] = cb
                yield cb

        yield Label("")  # spacer

        # Inputs will be added later in on_mount or via event
        self.inputs_container = Vertical(id="dynamic-inputs")
        yield self.inputs_container

        yield Label("")  # spacer

        # Prediction outputs
        for k in self.k_values:
            label = Label(
                "Predicted Diabetes: —",
                id=f"prediction-k{k}-{sanitize_id(self.model_name)}",
            )
            self.prediction_labels[k] = label
            yield label

    def on_mount(self) -> None:
        self._refresh_inputs()
        self.predict()

    @on(Checkbox.Changed)
    def on_checkbox_changed(self, event: Checkbox.Changed):
        feat = event.checkbox.label.plain
        if event.checkbox.value:
            self.selected_features.add(feat)
        else:
            self.selected_features.discard(feat)
        self._refresh_inputs()
        self.predict()

    @on(Input.Changed)
    def on_input_changed(self, event: Input.Changed):
        self.predict()

    def _refresh_inputs(self):
        print(f"Selected features: {sorted(self.selected_features)}")  # DEBUG
        self.inputs_container.remove_children()
        self.inputs.clear()
        for feat in self.all_feature_names:
            if feat not in self.selected_features:
                continue

            label = Label(f"{feat}:")
            validator = Number(minimum=0)  # All features ≥ 0
            safe_id = f"input-{sanitize_id(feat)}"
            input_widget = Input(
                placeholder=f"Enter {feat}",
                validators=[validator],
                id=safe_id,
            )
            if feat in self.initial_values:
                input_widget.value = str(self.initial_values[feat])
            self.inputs[feat] = input_widget
            self.inputs_container.mount(Horizontal(label, input_widget))

    def predict(self):
        if not self.selected_features:
            for k in self.k_values:
                self.prediction_labels[k].update(
                    f"Predicted Diabetes (k={k}): — (no features selected)"
                )
            return

        try:
            selected_list = sorted(
                self.selected_features, key=self.all_feature_names.index
            )
            input_values = []
            for feat in selected_list:
                val = self.inputs[feat].value.strip()
                if not val:
                    raise ValueError("Missing input")
                input_values.append(float(val))

            x_input = np.array(input_values)
            col_indices = [self.all_feature_names.index(f) for f in selected_list]
            x_data_selected = self.x_data_full[:, col_indices]

            predictions = predict_knn(
                x_input=x_input,
                x_data=x_data_selected,
                y_data=self.y_data,
                k_values=self.k_values,
            )

            for k, text in predictions.items():
                self.prediction_labels[k].update(text)

        except Exception:
            for k in self.k_values:
                self.prediction_labels[k].update("Predicted Diabetes: —")
