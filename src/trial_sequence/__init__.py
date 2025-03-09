import re
from functools import reduce
import warnings

import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import \
    PerfectSeparationWarning

from trial_sequence.calculate_weights import (calculate_censor_weights,
                                              calculate_switch_weights)
from trial_sequence.data_expansion import expand
from trial_sequence.data_manipulation import data_manipulation
from trial_sequence.utils import (te_data, te_expansion, te_outcome_data,
                                  te_outcome_model, te_stats_glm_logit,
                                  te_weights_spec, all_vars, add_rhs)

warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", PerfectSeparationWarning)


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

        self.expansion = None
        self.outcome_data = None
        self.outcome_model = None

    def __str__(self):
        if self.censor_weights is not None:
            censor_weights = str(self.censor_weights)
        else:
            censor_weights = " - No weight model specified"

        if self.switch_weights is not None:
            switch_weights = f"IPW for treatment switch censoring:\n {str(self.switch_weights)}\n"
        else:
            switch_weights = ""

        if self.data is not None:
            expansion = f"{self.expansion}\n"
        else:
            expansion = ""

        return f"""
Trial Sequence Object
Estimand: {trial_sequence.estimands[self.estimand]}

Data:
{self.data}

IPW for informative censoring:
{censor_weights}

{switch_weights}{expansion}

Outcome model:
{self.outcome_model}
        """

    def __repr__(self):
        return str(self)

    def set_data(
        self,
        data: pd.DataFrame,
        id="id",
        period="period",
        outcome="outcome",
        eligible="eligible",
        treatment="treatment",
    ) -> None:
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

    def update_outcome_formula(self) -> None:
        if self.outcome_model is None:
            return

        self.outcome_model.stabilised_weights_terms = self.get_stabilised_weights_terms()

        formula_list = [
            "1",
            self.outcome_model.treatment_terms,
            self.outcome_model.adjustment_terms,
            self.outcome_model.followup_time_terms,
            self.outcome_model.trial_period_terms,
            self.outcome_model.stabilised_weights_terms
        ]

        keep = [part for term in formula_list if term for part in term.split("~")]
        self.outcome_model.formula = f"outcome ~ {reduce(add_rhs, keep)}"

        adjustment_vars = set(all_vars(self.outcome_model.adjustment_terms) + 
                            all_vars(self.outcome_model.stabilised_weights_terms))

        self.outcome_model.adjustment_vars = list(adjustment_vars - {"1"})

    def set_switch_weight_model(
        self,
        numerator="1",
        denominator="1",
        model_fitter: te_stats_glm_logit = None,
        eligible_wts_0: str = None,
        eligible_wts_1: str = None,
    ) -> None:
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

        self.update_outcome_formula()

    def set_censor_weight_model(
        self,
        censor_event: str,
        numerator="1",
        denominator="1",
        pool_models: str = "none",
        model_fitter: te_stats_glm_logit = None,
    ) -> None:
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

        self.update_outcome_formula()

    def calculate_weights(self) -> None:
        if self.estimand == "PP" and self.switch_weights is None:
            raise ValueError(
                "Switch weight models are not specified. Use `set_switch_weight_model()`"
            )
        elif self.estimand == "AT" and self.censor_weights is None:
            raise ValueError(
                "Censor weight models are not specified. Use `set_censor_weight_model()`"
            )

        self.data.data["wt"] = 1

        if self.switch_weights is not None:
            if self.switch_weights is not None:
                calculate_switch_weights(self)
                self.data.data["wt"] *= self.data.data["wtS"]

        if self.censor_weights is not None:
            if self.censor_weights is not None:
                calculate_censor_weights(self)
                self.data.data["wt"] *= self.data.data["wtC"]

    def set_outcome_model(
        self,
        treatment_var="0",
        adjustment_terms="1",
        followup_time_terms="followup_time + I(followup_time**2)",
        trial_period_terms="trial_period + I(trial_period**2)",
        model_fitter=te_stats_glm_logit(save_path=None),
    ) -> None:
        if self.estimand == "ITT" or self.estimand == "PP":
            treatment_var = "assigned_treatment"
        else:
            treatment_var = "does"

        if not isinstance(self.data, te_data):
            raise ValueError("Use set_data() before set_outcome_model()")

        formula_list = {
            "treatment": treatment_var,
            "adjustment": adjustment_terms,
            "followup": followup_time_terms,
            "period": trial_period_terms,
            "stabilised": self.get_stabilised_weights_terms(),
        }

        adjustment = list(
            (
                set(all_vars(formula_list["adjustment"]))
                | set(all_vars(formula_list["stabilised"]))
            ) - {"1"}
        )

        if not set(adjustment).issubset(self.data.data.columns):
            raise ValueError("Variables in formulas must exist in dataset")

        treatment = treatment_var.split(" + ")

        self.outcome_model = te_outcome_model(
            treatment_var=treatment,
            adjustment_vars=adjustment,
            model_fitter=model_fitter,
            adjustment_terms=formula_list["adjustment"],
            treatment_terms=formula_list["treatment"],
            followup_time_terms=formula_list["followup"],
            trial_period_terms=formula_list["period"],
            stabilised_weights_terms=formula_list["stabilised"],
        )

        self.update_outcome_formula()

    def get_stabilised_weights_terms(self) -> str:
        stabilised_terms = "1"

        if self.censor_weights is not None:
            stabilised_terms += f" + {self.censor_weights.numerator}"

        if self.switch_weights is not None:
            stabilised_terms += f" + {self.switch_weights.numerator}"

        return stabilised_terms

    def set_expansion_options(
        self,
        output: object,
        chunk_size=0,
        first_period=0,
        last_period=float("inf"),
        censor_at_switch=False,
    ) -> None:
        if self.estimand == "PP":
            censor_at_switch = True

        if not isinstance(output, object):
            raise TypeError(
                "Expected output to be an instance of te_datastore"
            )

        if not isinstance(chunk_size, int) or chunk_size < 0:
            raise ValueError("chunk_size must be a non-negative integer")

        if first_period != 0 and not isinstance(first_period, int):
            raise ValueError("first_period must be an integer")

        if last_period != float("inf") and not isinstance(last_period, int):
            raise ValueError("last_period must be an integer")

        self.expansion = te_expansion(
            chunk_size=chunk_size,
            datastore=output,
            first_period=first_period,
            last_period=last_period,
            censor_at_switch=censor_at_switch,
        )

    def expand_trials(self) -> None:
        if self.expansion is None:
            raise ValueError(
                "Use set_expansion_options() before expand_trials()"
            )

        if self.data is None:
            raise ValueError("Use set_data() before expand_trials()")

        data = self.data.data.copy()

        eligible = self.data.data["eligible"] == 1
        first_period = max(
            self.expansion.first_period,
            self.data.data.loc[eligible, "period"].min(),
        )
        last_period = min(
            self.expansion.last_period,
            self.data.data.loc[eligible, "period"].max(),
        )
        chunk_size = self.expansion.chunk_size
        censor_at_switch = self.expansion.censor_at_switch

        outcome_adj_vars = list(set(self.outcome_model.adjustment_vars))
        keeplist = list(
            set(
                [
                    "id",
                    "trial_period",
                    "followup_time",
                    "outcome",
                    "weight",
                    "treatment",
                ]
                + outcome_adj_vars
                + self.outcome_model.treatment_var
            )
        )

        if "wt" not in data.columns:
            data["wt"] = 1

        all_ids = data["id"].unique()

        if chunk_size == 0:
            ids_split = [all_ids]
        else:
            ids_split = np.array_split(
                all_ids, np.ceil(len(all_ids) / chunk_size)
            )

        for ids in ids_split:
            switch_data = expand(
                sw_data=data[data["id"].isin(ids)],
                outcomeCov_var=outcome_adj_vars,
                where_var=None,
                use_censor=censor_at_switch,
                minperiod=first_period,
                maxperiod=last_period,
                keeplist=keeplist,
            )

            self.expansion.datastore.save_expanded_data(switch_data)

    def load_expanded_data(
        self,
        p_control: float,
        period: int = None,
        subset_condition: str = None,
        seed: int = None,
    ) -> None:
        assert self.expansion.datastore.N > 0, "N must be positive"
        assert p_control is None or (
            0 <= p_control <= 1
        ), "p_control must be between 0 and 1"
        assert (
            period is None or isinstance(period, int) and period >= 0
        ), "period must be a non-negative integer"

        if subset_condition is not None:
            assert isinstance(
                subset_condition, str
            ), "subset_condition must be a string"

        assert seed is None or (
            isinstance(seed, int) and seed >= 0
        ), "seed must be a non-negative integer"

        if p_control is None:
            data_table = self.expansion.datastore.read_expanded_data(
                self.expansion.datastore,
                period=period,
                subset_condition=subset_condition,
            )
            data_table["sample_weight"] = 1
        else:
            data_table = self.expansion.datastore.sample_expanded_data(
                period=period,
                subset_condition=subset_condition,
                p_control=p_control,
                seed=seed,
            )

        self.outcome_data = te_outcome_data(
            data_table, p_control, subset_condition
        )

    def fit_msm(self, weight_cols:list=None, modify_weights:callable=None) -> None:
        if self.outcome_model is None:
            raise ValueError("Outcome model not set, please run set_outcome_model")
        if self.expansion.datastore.N == 0:
            raise ValueError("Datastore is empty, please run expand_trials")
        if not hasattr(self.outcome_data, 'n_rows') or self.outcome_data.n_rows == 0:
            raise ValueError("Outcome data is empty, please run load_expanded_data")

        data = self.outcome_data.data

        if weight_cols:
            if not set(weight_cols).issubset(data.columns):
                raise ValueError("Some weight columns are missing from outcome_data")
            data["w"] = data[weight_cols].prod(axis=1)
        else:
            data["w"] = 1

        if modify_weights:
            temp_weights = modify_weights(data["w"])
            if not np.all((temp_weights >= 0) & np.isfinite(temp_weights)):
                raise ValueError("Modified weights must be non-negative and finite")
            data["w"] = temp_weights

        self.outcome_model.fitted = self.outcome_model.model_fitter.fit_outcome_model(
            data=data,
            formula=self.outcome_model.formula,
            weights=data["w"]
        )

    def predict(self, newdata, predict_times, conf_int=True, samples=100, type='cum_inc'):
        return self.outcome_model.fitted.predict(
            newdata=newdata,
            predict_times=predict_times,
            conf_int=conf_int,
            samples=samples,
            type=type
        )