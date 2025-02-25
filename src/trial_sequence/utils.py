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

    def __repr__(self):
        numerator = (
            f"Numerator formula: {self.numerator}\n"
            if self.numerator is not None
            else ""
        )
        denominator = (
            f"Denominator formula: {self.denominator}\n"
            if self.denominator is not None
            else ""
        )
        pool = (
            f"Numerator model is {"pooled across treatment arms" if self.pool_numerator else "not pooled"}. Denominator model is {"pooled" if self.pool_denominator else 'not pooled'}\n"
            if self.pool_numerator or self.pool_denominator
            else ""
        )
        model_fitter = (
            f"Model fitter type: {self.model_fitter.__class__.__name__}\n"
            if self.model_fitter is not None
            else ""
        )

        return f"{numerator}{denominator}{pool}{model_fitter}"


class te_weights_fitted:
    def __init__(self, **kwargs):
        self.label = kwargs.get("label", None)
        self.summary = kwargs.get("summary", None)
        self.fitted = kwargs.get("fitted", None)
