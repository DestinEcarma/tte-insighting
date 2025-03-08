import os
import tempfile

import joblib
import pandas as pd
import statsmodels.api as sm

from trial_sequence.utils import te_stats_glm_logit, te_weights_fitted


def fit_weights_model(
    object: te_stats_glm_logit, data: pd.DataFrame, formula: str, label: str
) -> te_weights_fitted:
    model = sm.GLM.from_formula(
        formula,
        data,
        family=sm.families.Binomial(link=sm.families.links.Logit()),
    ).fit()
    file_path = None

    if object.save_path is not None:
        os.makedirs(object.save_path, exist_ok=True)
        file_path = tempfile.NamedTemporaryFile(
            prefix="model_",
            dir=object.save_path,
            delete=False,
            suffix=".joblib",
        ).name
        joblib.dump(model, file_path)

    return te_weights_fitted(
        label=label,
        summary={
            "tidy": model.summary2().tables[1],
            "glance": model.summary2().tables[0],
            "save_path": (
                pd.DataFrame({"path": [file_path]}) if file_path else None
            ),
        },
        fitted=model.fittedvalues,
    )
