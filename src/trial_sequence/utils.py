import os
import re
import warnings
import numpy as np
import tempfile
import pandas as pd
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy.stats import multivariate_normal
from trial_sequence.predict import check_newdata, calculate_survival, calculate_cum_inc, calculate_predictions

class te_data:
    data: pd.DataFrame
    nobs: int
    n: int

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


class te_model_fitter:
    save_path: str

    def __init__(self, save_path: str):
        self.save_path = save_path


class te_stats_glm_logit(te_model_fitter):
    def __init__(self, save_path: str):
        super().__init__(save_path)

    def fit_outcome_model(self, data, formula, weights=None):
        if weights is None:
            data['weights'] = 1
        else:
            data['weights'] = weights

        model = glm(formula=formula, data=data, family=Binomial()).fit(weights=data['weights'])

        save_path = None

        if self.save_path is not None:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)

            file = tempfile.NamedTemporaryFile(dir=self.save_path, prefix="model_", suffix=".pkl", delete=False)
            model.save(file.name)
            save_path = pd.DataFrame({'save': [file.name]})

        vcov = cov_cluster(model, group=data['id'])

        model_list = {
            'model': model,
            'vcov': vcov
        }

        summary = {
            'tidy': model.summary(),
            'glance': {
                'params': model.params,
                'pvalues': model.pvalues,
                'rsquared_mcfadden': 1 - (model.deviance / model.null_deviance)
            }
        }

        if self.save_path is not None:
            summary['save_path'] = save_path

        return te_stats_glm_logit_outcome_fitted(model=model_list, summary=summary)


class te_outcome_fitted:
    model: dict
    summary: dict

    def __init__(self, **kwargs):
        self.model = kwargs.get("model", None)
        self.summary = kwargs.get("summary", None)

    def __str__(self):
        if len(self.summary) > 0:
            return f"""Model Summary:

{self.summary['tidy']}

{self.summary['glance']}
            """
        else:
            return "Use fit_msm() to fit the outcome model"

class te_stats_glm_logit_outcome_fitted(te_outcome_fitted):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, newdata, predict_times, conf_int=True, samples=100, type='cum_inc'):
        type = type.lower()

        if type not in ['cum_inc', 'survival']:
            raise ValueError("Type must be 'cum_inc' or 'survival'.")

        if not isinstance(predict_times, (list, np.ndarray)) or any(t < 0 for t in predict_times):
            raise ValueError("predict_times must be a list or array of non-negative integers.")

        if not isinstance(conf_int, bool):
            raise ValueError("conf_int must be a boolean.")

        if not isinstance(samples, int) or samples < 1:
            raise ValueError("samples must be a positive integer.")

        coefs_mat = np.array(self.model['model'].params).reshape(1, -1)

        if conf_int:
            if self.vcov.shape[0] != coefs_mat.shape[1] or self.vcov.shape[1] != coefs_mat.shape[1]:
                raise ValueError("Variance-covariance matrix dimensions do not match coefficients.")

            coefs_mat = np.vstack([
                coefs_mat,
                multivariate_normal.rvs(mean=self.model.params, cov=self.vcov, size=samples)
            ])

        newdata = check_newdata(newdata, self.model, predict_times)

        pred_fun = calculate_survival if type == "survival" else calculate_cum_inc

        pred_list = calculate_predictions(
            newdata=newdata,
            model=self.model,
            treatment_values={'assigned_treatment_0': 0, 'assigned_treatment_1': 1},
            pred_fun=pred_fun,
            coefs_mat=coefs_mat,
            matrix_n_col=len(predict_times)
        )

        pred_list['difference'] = pred_list['assigned_treatment_1'] - pred_list['assigned_treatment_0']

        results = {}
        for col_name in ['assigned_treatment_0', 'assigned_treatment_1', 'difference']:
            pred_matrix = pred_list[col_name]
            if conf_int:
                quantiles = np.quantile(pred_matrix, [0.025, 0.975], axis=0)
                results[col_name] = pd.DataFrame({
                    'followup_time': predict_times,
                    col_name: pred_matrix[0],
                    '2.5%': quantiles[0],
                    '97.5%': quantiles[1]
                })
            else:
                results[col_name] = pd.DataFrame({
                    'followup_time': predict_times,
                    col_name: pred_matrix[0]
                })

        return results

class te_weights_spec:
    numerator: str
    denominator: str
    pool_numerator: bool
    pool_denominator: bool
    model_fitter: te_model_fitter
    fitted: dict
    data_subset_expr: dict

    def __init__(self, **kwargs):
        self.numerator = kwargs.get("numerator", None)
        self.denominator = kwargs.get("denominator", None)
        self.pool_numerator = kwargs.get("pool_numerator", False)
        self.pool_denominator = kwargs.get("pool_denominator", False)
        self.model_fitter = kwargs.get("model_fitter", None)
        self.fitted = kwargs.get("fitted", {})
        self.data_subset_expr = kwargs.get("data_subset_expr", {})

    def __str__(self):
        numerator = f" - Numerator formula: {self.numerator}\n"
        denominator = f" - Denominator formulat: {self.denominator}\n"

        if self.pool_numerator:
            if self.pool_denominator:
                pool = " - Numerator and denominotor models are pooled across treatment arms.\n"
            else:
                pool = " - Numerator model is pooled across treatment arms. Denominator model is not pooled.\n"
        else:
            pool = ""

        model_fitter = (
            f" - Model fitter type: {self.model_fitter.__class__.__name__}\n"
        )

        if len(self.fitted) > 0:
            fitted = " - View weight model summaries with show_weight_models()"
        else:
            fitted = " - Weight models not fitted. Use calculate_weights()"

        return f"{numerator}{denominator}{pool}{model_fitter}{fitted}"

    def __repr__(self):
        return str(self)


class te_weights_fitted:
    label: str
    summary: dict
    fitted: float

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


class te_outcome_fitted:
    model: dict
    summary: dict

    def __init__(self, **kwargs):
        self.model = kwargs.get("model", None)
        self.summary = kwargs.get("summary", None)

    def __str__(self):
        if len(self.summary):
            summary = f"Model Summary:\n\n"
            summary += f"{self.summary["tidy"]}\n\n"
            summary += f"{self.summary["glance"]}"

            return summary
        else:
            return "Use fit_msm() to fit the outcome modle"


class te_outcome_model:
    formula: str
    adjustment_vars: str
    treatment_var: str
    adjustment_terms: str
    treatment_terms: str
    followup_time_terms: str
    trial_period_terms: str
    stabilised_weights_terms: str
    model_fitter: te_model_fitter
    fitted: te_outcome_fitted

    def __init__(self, **kwargs):
        self.formula = kwargs.get("formula", None)
        self.adjustment_vars = kwargs.get("adjustment_vars", None)
        self.treatment_var = kwargs.get("treatment_var", None)
        self.adjustment_terms = kwargs.get("adjustment_terms", None)
        self.treatment_terms = kwargs.get("treatment_terms", None)
        self.followup_time_terms = kwargs.get("followup_time_terms", None)
        self.trial_period_terms = kwargs.get("trial_period_terms", None)
        self.stabilised_weights_terms = kwargs.get(
            "stabilised_weights_terms", None
        )
        self.model_fitter = kwargs.get("model_fitter", None)
        self.fitted = kwargs.get("fitted", None)

    def __str__(self):
        formula = f"- Formula: {self.formula}\n"
        treatment_var = f"- Treatment variable: {self.treatment_var}\n"
        adjustment_vars = f"- Adjustment variables: {self.adjustment_vars}\n"
        model_fitter = (
            f"- Model fitter type: {self.model_fitter.__class__.__name__}\n\n"
        )

        return f"{formula}{treatment_var}{adjustment_vars}{model_fitter}{self.fitted}"
    
    def __repr__(self):
        return str(self)


class te_datastore:
    N: int

    def __init__(self, N=0):
        self.N = N

    def sample_expanded_data(
        self,
        p_control: float,
        period: int = None,
        subset_condition: str = None,
        seed: int = None,
    ):
        old_seed = np.random.get_state()

        try:
            np.random.seed(seed)
            data = self.read_expanded_data(
                period=period, subset_condition=subset_condition
            )

            sampled_data = (
                data.groupby(
                    ["trial_period", "followup_time"], group_keys=False
                )
                .apply(lambda df: do_sampling(df, p_control))
                .reset_index(drop=True)
            )

            return sampled_data
        finally:
            np.random.set_state(old_seed)


class te_datastore_datatable(te_datastore):
    data: pd.DataFrame

    def __init__(self, data=pd.DataFrame(), N=0):
        super().__init__(N)
        self.data = data

    def __str__(self):
        return f"""A TE Datastore Datatable object
N: {self.N} observations
{self.data}
        """

    def save_expanded_data(self, data: pd.DataFrame) -> None:
        self.data = pd.concat([self.data, data], ignore_index=True)
        self.N = len(self.data)

    def read_expanded_data(
        self, period: int = None, subset_condition: str = None
    ):
        if period is not None:
            if not isinstance(period, (int, list)):
                raise ValueError(
                    "period must be an integer or a list of integers."
                )
            if isinstance(period, int):
                period = [period]

        if period is None:
            data_table = self.data.copy()
        else:
            data_table = self.data[
                self.data["trial_period"].isin(period)
            ].copy()

        if subset_condition:
            data_table = data_table.query(subset_condition)

        return data_table


class te_expansion:
    chunk_size: int
    datastore: te_datastore
    censor_at_switch: bool
    first_period: int
    last_period: int

    def __init__(self, **kwargs):
        self.chunk_size = kwargs.get("chunk_size", None)
        self.datastore = kwargs.get("datastore", None)
        self.censor_at_switch = kwargs.get("censor_at_switch", None)
        self.first_period = kwargs.get("first_period", None)
        self.last_period = kwargs.get("last_period", None)

    def __str__(self):
        if self.datastore.N > 0:
            datastore = f"\n{self.datastore}"
        else:
            datastore = "- Use expand_trials() to construct sequence of trials dataset."

        return f"""Sequence of Trials Data:
- Chunk Size: {self.chunk_size}
- Censor at switch: {self.censor_at_switch}
- First period: {self.first_period} | Last period: {self.last_period}
{datastore}
        """

    def __repr__(self):
        return str(self)


class te_outcome_data:
    data: pd.DataFrame
    n_rows: int
    n_ids: int
    periods: int
    p_control: float
    subset_condition: str

    def __init__(
        self, data, p_control: float = None, subset_condition: str = None
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        required_columns = {
            "id",
            "trial_period",
            "followup_time",
            "outcome",
            "weight",
        }
        if not required_columns.issubset(data.columns):
            raise ValueError(
                f"Missing required columns: {required_columns - set(data.columns)}"
            )

        n_rows = len(data)
        if n_rows == 0:
            warnings.warn("Outcome data has 0 rows")

        n_ids = data["id"].nunique()
        periods = sorted(data["trial_period"].unique().tolist())

        subset_condition = subset_condition or []
        p_control = p_control or []

        self.data = data
        self.n_rows = n_rows
        self.n_ids = n_ids
        self.periods = periods
        self.p_control = p_control
        self.subset_condition = subset_condition


def stats_glm_logit(save_path: str) -> te_stats_glm_logit:
    os.makedirs(save_path, exist_ok=True)

    return te_stats_glm_logit(save_path)


def save_to_datatable() -> te_datastore_datatable:
    return te_datastore_datatable()


def do_sampling(data: pd.DataFrame, p_control: float = 0.01) -> pd.DataFrame:
    cases = data[data["outcome"] == 1].index.to_list()
    controls = data[data["outcome"] == 0].index.to_list()

    n_cases = len(cases)
    n_controls = len(controls)

    n_sample = np.random.binomial(n_controls, p_control)
    control_select = (
        np.random.choice(controls, size=n_sample, replace=False)
        if n_sample > 0
        else []
    )

    sampled_indices = np.concatenate((cases, control_select))
    sampled_data = data.loc[sampled_indices].copy()

    sampled_data["sample_weight"] = np.concatenate(
        (
            np.ones(n_cases),
            np.ones(n_sample) / p_control if n_sample > 0 else [],
        )
    )

    return sampled_data

def all_vars(formula: str) -> list:
    return re.findall(r'(?<![a-zA-Z0-9_])([a-zA-Z_][a-zA-Z0-9_]*)(?![a-zA-Z0-9_])', formula)

def add_rhs(formula1: str, formula2: str) -> str:
    return f"{formula1} + {formula2}"