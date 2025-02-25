import pandas as pd

from trial_sequence.calculate_weights import (calculate_censor_weights,
                                              calculate_switch_weights)
from trial_sequence.data_manipulation import data_manipulation
from trial_sequence.utils import te_data, te_stats_glm_logit, te_weights_spec


class trial_sequence:
    estimands = {
        "AT": "As-Treated",
        "PP": "Per-Protocol",
        "ITT": "Intent-to-Treat",
    }

    def __init__(self, estimand: str, **kwargs):
        if estimand.upper() not in self.estimands.keys():
            raise ValueError(
                f"Estimand must be one of {', '.join(self.estimands.keys())}"
            )

        self.estimand = estimand.upper()
        self.data = kwargs.get("data", None)
        self.id = kwargs.get("id", None)
        self.period = kwargs.get("period", None)
        self.outcome = kwargs.get("outcome", None)
        self.eligible = kwargs.get("eligible", None)
        self.treatment = kwargs.get("treatment", None)

        self.switch_weights = None
        self.censor_weights = None

    def __repr__(self):
        return f"""
Trial Sequence Object
Estimand: {trial_sequence.estimands[self.estimand]}

Data:
{self.data.data.head() if self.data is not None else None}
        """

    def set_data(
        self,
        data: pd.DataFrame,
        id="id",
        period="period",
        outcome="outcome",
        eligible="eligible",
        treatment="treatment",
    ) -> "trial_sequence":
        if self.estimand == "ITT" or self.estimand == "AT":
            censor_at_switch = False
        elif self.estimand == "PP":
            censor_at_switch = True

        data_cols = [id, period, treatment, outcome, eligible]
        new_col_names = ["id", "period", "treatment", "outcome", "eligible"]

        missing_cols = set(data_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in data: {', '.join(missing_cols)}"
            )

        invalid_cols = {
            "wt",
            "wtC",
            "weight",
            "dose",
            "assigned_treatment",
        } & set(data.columns)
        if invalid_cols:
            raise ValueError(
                f"Invalid column names in data: {', '.join(invalid_cols)}"
            )

        if len(set(data_cols)) != len(data_cols):
            duplicates = [col for col in data_cols if data_cols.count(col) > 1]
            raise ValueError(
                f"Duplicate column names specified: {', '.join(duplicates)}"
            )

        trial_data = data.rename(columns=dict(zip(data_cols, new_col_names)))
        trial_data = data_manipulation(trial_data, use_censor=censor_at_switch)

        self.data = te_data(
            data=trial_data, nobs=len(trial_data), n=trial_data["id"].nunique()
        )

        return self

    def set_switch_weight_model(
        self,
        numerator="1",
        denominator="1",
        model_fitter: te_stats_glm_logit = None,
        eligible_wts_0: str = None,
        eligible_wts_1: str = None,
    ) -> "trial_sequence":
        if self.estimand == "ITT":
            raise ValueError(
                "Switching weights are not supported for intention-to-treat analysis"
            )

        if not isinstance(self.data, te_data):
            raise ValueError(
                "Please use `set_data()` to set up the data before setting switch weight models"
            )

        cols = self.data.data.columns.tolist()

        if eligible_wts_0 is not None:
            if eligible_wts_0 not in cols:
                raise ValueError(
                    f"Column `{eligible_wts_0}` must be included in the data"
                )

            self.data.rename(
                columns={eligible_wts_0: "eligible_wts_0"}, inplace=True
            )

        if eligible_wts_1 is not None:
            if eligible_wts_1 not in cols:
                raise ValueError(
                    f"Column `{eligible_wts_1}` must be included in the data"
                )

            self.data.rename(
                columns={eligible_wts_1: "eligible_wts_1"}, inplace=True
            )

        if "time_on_regime" in numerator:
            raise ValueError("time_on_regime should not be used in numerator")

        numerator = f"treatment ~ {numerator}"
        denominator = f"treatment ~ {denominator}"

        self.switch_weights = te_weights_spec(
            numerator=numerator,
            denominator=denominator,
            model_fitter=model_fitter,
        )

        # Too lazy to implement this function
        # The tutorial does not realy use this
        # self.update_outcome_formula()
        return self

    def set_censor_weight_model(
        self,
        censor_event: str,
        numerator="1",
        denominator="1",
        pool_models: str = "none",
        model_fitter: te_stats_glm_logit = None,
    ) -> "trial_sequence":
        if "time_on_regime" in numerator:
            raise ValueError("time_on_regime should not be used in numerator")

        if not isinstance(self.data, te_data):
            raise ValueError(
                "Please use `set_data()` to set up the data before setting censor weight models"
            )

        cols = self.data.data.columns.tolist()

        if censor_event not in cols:
            raise ValueError(
                f"Column `{censor_event}` must be included in the data"
            )

        numerator = f"1 - {censor_event} ~ {numerator}"
        denominator = f"1 - {censor_event} ~ {denominator}"

        if self.estimand == "PP" or self.estimand == "AT":
            if pool_models not in ["numerator", "both", None]:
                raise ValueError(
                    "Pool models must be one of `both`, `numerator`, or `None`"
                )
        elif self.estimand == "ITT":
            if pool_models not in ["numerator", "both"]:
                raise ValueError(
                    "Pool models must be one of `both` or `numerator`"
                )

        pool_numerator = pool_models in ["numerator", "both"]
        pool_denominator = pool_models == "both"

        self.censor_weights = te_weights_spec(
            numerator=numerator,
            denominator=denominator,
            pool_numerator=pool_numerator,
            pool_denominator=pool_denominator,
            model_fitter=model_fitter,
        )

        # Too lazy to implement this function
        # The tutorial does not realy use this
        # self.update_outcome_formula()
        return self

    def calculate_weights(self):
        if self.estimand == "PP" and self.switch_weights is None:
            raise ValueError(
                "Switch weight models are not specified. Use `set_switch_weight_model()`"
            )
        elif self.estimand == "AT" and self.censor_weights is None:
            raise ValueError(
                "Censor weight models are not specified. Use `set_censor_weight_model()`"
            )

        self.data.wt = 1

        if self.switch_weights is not None:
            if self.switch_weights is not None:
                self = calculate_switch_weights(self)
                self.data.wt *= self.data.wtS

        if self.censor_weights is not None:
            if self.censor_weights is not None:
                self = calculate_censor_weights(self)
                self.data.wt *= self.data.wtC

        return self
