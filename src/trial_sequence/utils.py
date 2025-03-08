import os

import pandas as pd


def stats_glm_logit(save_path: str) -> None:
    os.makedirs(save_path, exist_ok=True)

    return te_stats_glm_logit(save_path)


class te_data:
    def __init__(self, data: pd.DataFrame, nobs: int, n: int):
        self.data = data
        self.nobs = nobs
        self.n = n

    def __str__(self):
        n = f" - N: {self.nobs} observations from {self.n} patients\n"
        data = self.data[
            list(
                set(self.data.columns)
                - set(
                    [
                        "time_of_event",
                        "first",
                        "am_1",
                        "cumA",
                        "switch",
                        "regime_start",
                        "eligible0",
                        "eligible1",
                        "p_n",
                        "p_d",
                        "pC_n",
                        "pC_d",
                    ]
                )
            )
        ]

        return f"{n}{data}"


class te_stats_glm_logit:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.model_spec = None


class te_weights_spec:
    def __init__(self, **kwargs):
        self.numerator = kwargs.get("numerator", None)
        self.denominator = kwargs.get("denominator", None)
        self.pool_numerator = kwargs.get("pool_numerator", False)
        self.pool_denominator = kwargs.get("pool_denominator", False)
        self.model_fitter = kwargs.get("model_fitter", None)
        self.fitted = kwargs.get("fitted", {})
        self.data_subset_expr = kwargs.get("data_subset_expr", {})

    def __str__(self):
        numerator = f"Numerator formula: {self.numerator}\n"
        denominator = f"Denominator formulat: {self.denominator}\n"

        if self.pool_numerator:
            if self.pool_denominator:
                pool = "Numerator and denominotor models are pooled across treatment arms.\n"
            else:
                pool = "Numerator model is pooled across treatment arms. Denominator model is not pooled.\n"
        else:
            pool = ""

        model_fitter = (
            f"Model fitter type: {self.model_fitter.__class__.__name__}\n"
        )

        if len(self.fitted) > 0:
            fitted = "View weight model summaries with show_weight_models()\n"
        else:
            fitted = "Weight models not fitted. Use calculate_weights()\n"

        return f"{numerator}{denominator}{pool}{model_fitter}{fitted}"

    def __repr__(self):
        return str(self)


class te_weights_fitted:
    def __init__(self, **kwargs):
        self.label = kwargs.get("label", None)
        self.summary = kwargs.get("summary", None)
        self.fitted = kwargs.get("fitted", None)

    def __str__(self):
        model = f"Model: {self.label}\n\n"
        summary = ""

        for df in self.summary.values():
            summary += f"{df.to_string(index=False)}\n\n"

        return f"{model}{summary}"
