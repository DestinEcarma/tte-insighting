import os
import tempfile

import joblib
import pandas as pd
import parsnip

from trial_sequence.utils import te_stats_glm_logit, te_weights_fitted


def fit_weights_model(
    object: te_stats_glm_logit, data: pd.DataFrame, formula: str, label: str
) -> te_weights_fitted:
    data["treatment"] = pd.Categorical(data["treatment"], categories=[0, 1])

    # FIXME: `parsnip.fit()` is not a function in the `astro-parsnip` package
    parsnip_fit = parsnip.fit(object.model_spec, formula, data=data)

    if object.save_path is not None:
        if not os.path.exists(object.save_path):
            os.makedirs(object.save_path, exist_ok=True)
        file_path = tempfile.NamedTemporaryFile(
            prefix="model_",
            dir=object.save_path,
            delete=False,
            suffix=".joblib",
        ).name
        joblib.dump(parsnip_fit, file_path)

    fitted = parsnip_fit.predict_proba(data)[:, 1]

    try:
        tidy = pd.DataFrame(parsnip_fit.coef_, columns=["coefficient"])
    except Exception as e:
        tidy = pd.DataFrame({"error": str(e)})

    try:
        glance = pd.DataFrame(parsnip_fit.score(data))
    except Exception as e:
        glance = pd.DataFrame({"error": str(e)})

    return te_weights_fitted(
        label=label,
        summary={
            "tidy": tidy,
            "glance": glance,
            "save_path": pd.DataFrame({"path": [file_path]}),
        },
        fitted=fitted,
    )


def calculate_switch_weights(object):
    if "eligible_wts_1" in object.data.data.columns:
        data_1_expr = (object.data.data["am_1"] == 1) & (
            object.data.data["eligible_wts_1"] == 1
        )
    else:
        data_1_expr = object.data.data["am_1"] == 1

    model_1_index = object.data.data[data_1_expr].index

    object.switch_weights.fitted.n1 = fit_weights_model(
        object=object.switch_weights.model_fitter,
        data=object.data.data.loc[model_1_index],
        formula=object.switch_weights.numerator,
        label="P(treatment = 1 | previous treatment = 1) for numerator",
    )
    object.data.data.loc[model_1_index, "p_n"] = (
        object.switch_weights.fitted.n1.fitted
    )
    object.switch_weights.data_subset_expr["n1"] = data_1_expr

    object.switch_weights.fitted.d1 = fit_weights_model(
        object=object.switch_weights.model_fitter,
        data=object.data.data.loc[model_1_index],
        formula=object.switch_weights.denominator,
        label="P(treatment = 1 | previous treatment = 1) for denominator",
    )
    object.data.data.loc[model_1_index, "p_d"] = (
        object.switch_weights.fitted.d1.fitted
    )
    object.switch_weights.data_subset_expr["d1"] = data_1_expr
    del model_1_index

    if "eligible_wts_1" in object.data.data.columns:
        data_0_expr = (object.data.data["am_1"] == 0) & (
            object.data.data["eligible_wts_1"] == 0
        )
    else:
        data_0_expr = object.data.data["am_1"] == 0

    model_0_index = object.data.data[data_0_expr]

    object.switch_weights.fitted.n0 = fit_weights_model(
        object=object.switch_weights.model_fitter,
        data=object.data.data.loc[model_0_index],
        formula=object.switch_weights.numerator,
        label="P(treatment = 1 | previous treatment = 0) for numerator",
    )
    object.data.data.loc[model_0_index, "p_n"] = (
        object.switch_weights.fitted.n0.fitted
    )
    object.switch_weights.data_subset_expr["n0"] = data_0_expr

    object.switch_weights.fitted.d0 = fit_weights_model(
        object=object.switch_weights.model_fitter,
        data=object.data.data.loc[model_0_index],
        formula=object.switch_weights.denominator,
        label="P(treatment = 1 | previous treatment = 0) for denominator",
    )
    object.data.data.loc[model_0_index, "p_d"] = (
        object.switch_weights.fitted.d0.fitted
    )
    object.switch_weights.data_subset_expr["d0"] = data_0_expr
    del model_0_index

    if any(
        col in object.data.data.columns
        for col in ["eligible_wts_0", "eligible_wts_1"]
    ):
        object.data.data.loc[
            (
                (object.data.data["eligible_wts_0"] == 1)
                | (object.data.data["eligible_wts_1"] == 1)
            )
            & (object.data.data["treatment"] == 0),
            "wtS",
        ] = (1.0 - object.data.data["p_n"]) / (1.0 - object.data.data["p_d"])

        object.data.data.loc[
            (
                (object.data.data["eligible_wts_0"] == 1)
                | (object.data.data["eligible_wts_1"] == 1)
            )
            & (object.data.data["treatment"] == 1),
            "wtS",
        ] = (
            object.data.data["p_n"] / object.data.data["p_d"]
        )
    else:
        object.data.data.loc[object.data.data["treatment"] == 0, "wtS"] = (
            1.0 - object.data.data["p_n"]
        ) / (1.0 - object.data.data["p_d"])
        object.data.data.loc[object.data.data["treatment"] == 1, "wtS"] = (
            object.data.data["p_n"] / object.data.data["p_d"]
        )

    object.data.data["wtS"].ffill(1, inplace=True)

    return object


def calculate_censor_weights(object):
    if (
        not object.censor_weights.pool_numerator
        or not object.censor_weights.pool_denominator
    ):
        data_0_expr = object.data.data["am_1"] == 0
        data_1_expr = object.data.data["am_1"] == 1
        elig_0_index = object.data.data.index[data_0_expr]
        elig_1_index = object.data.data.index[data_1_expr]

    data_pool_expr = True

    if object.censor_weights.pool_numerator:
        object.censor_weights.fitted.n = fit_weights_model(
            object=object.censor_weights.model_fitter,
            data=object.data.data,
            formula=object.censor_weights.numerator,
            label="P(censor_event = 0 | X) for numerator",
        )
        object.data.data["pC_n"] = object.censor_weights.fitted.n.fitted
        object.censor_weights.data_subset_expr["n"] = data_pool_expr
    else:
        object.censor_weights.fitted.n0 = fit_weights_model(
            object=object.censor_weights.model_fitter,
            data=object.data.data.loc[elig_0_index],
            formula=object.censor_weights.numerator,
            label="P(censor_event = 0 | X, previous treatment = 0) for numerator",
        )
        object.data.data.loc[elig_0_index, "pC_n"] = (
            object.censor_weights.fitted.n0.fitted
        )
        object.censor_weights.data_subset_expr["n0"] = data_0_expr

        object.censor_weights.fitted.n1 = fit_weights_model(
            object=object.censor_weights.model_fitter,
            data=object.data.data.loc[elig_1_index],
            formula=object.censor_weights.numerator,
            label="P(censor_event = 0 | X, previous treatment = 1) for numerator",
        )
        object.data.data.loc[elig_1_index, "pC_n"] = (
            object.censor_weights.fitted.n1.fitted
        )
        object.censor_weights.data_subset_expr["n1"] = data_1_expr

    if object.censor_weights.pool_denominator:
        object.censor_weights.fitted.d = fit_weights_model(
            object=object.censor_weights.model_fitter,
            data=object.data.data,
            formula=object.censor_weights.denominator,
            label="P(censor_event = 0 | X) for denominator",
        )
        object.data.data["pC_d"] = object.censor_weights.fitted.d.fitted
        object.censor_weights.data_subset_expr["d"] = data_pool_expr
    else:
        object.censor_weights.fitted.d0 = fit_weights_model(
            object=object.censor_weights.model_fitter,
            data=object.data.data.loc[elig_0_index],
            formula=object.censor_weights.denominator,
            label="P(censor_event = 0 | X, previous treatment = 0) for denominator",
        )
        object.data.data.loc[elig_0_index, "pC_d"] = (
            object.censor_weights.fitted.d0.fitted
        )
        object.censor_weights.data_subset_expr["d0"] = data_0_expr

        object.censor_weights.fitted.d1 = fit_weights_model(
            object=object.censor_weights.model_fitter,
            data=object.data.data.loc[elig_1_index],
            formula=object.censor_weights.denominator,
            label="P(censor_event = 0 | X, previous treatment = 1) for denominator",
        )
        object.data.data.loc[elig_1_index, "pC_d"] = (
            object.censor_weights.fitted.d1.fitted
        )
        object.censor_weights.data_subset_expr["d1"] = data_1_expr

    object.data.data["pC_d"].ffill(1, inplace=True)
    object.data.data["pC_n"].ffill(1, inplace=True)
    object.data.data["wtC"] = (
        object.data.data["pC_n"] / object.data.data["pC_d"]
    )

    return object
