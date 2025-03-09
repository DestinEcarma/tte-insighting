import pandas as pd
import numpy as np

def check_newdata(newdata, model, predict_times):
    required_vars = [var for var in model['formula'] if var != 'outcome']
    
    if newdata is None:
        newdata = model['data'][required_vars][model['data']['followup_time'] == 0]
    else:
        if not isinstance(newdata, pd.DataFrame) or newdata.shape[0] < 1:
            raise ValueError("newdata must be a DataFrame with at least one row.")
        
        missing_vars = set(required_vars) - set(newdata.columns)
        if missing_vars:
            raise ValueError(f"newdata must include the following columns: {', '.join(missing_vars)}")
        
        newdata = newdata[required_vars]
        newdata = newdata[newdata['followup_time'] == 0]

        col_attr_model = {var: model['data'][var].dtype for var in required_vars}
        col_attr_newdata = {var: newdata[var].dtype for var in required_vars}

        if col_attr_model != col_attr_newdata:
            print("Attributes of newdata do not match data used for fitting. Attempting to fix.")
            newdata = pd.concat([model['data'][required_vars].iloc[0:0], newdata], ignore_index=True)
            fixed = {var: model['data'][var].dtype == newdata[var].dtype for var in required_vars}
            if not all(fixed.values()):
                print(fixed)
                raise ValueError("Attributes do not match.")

    n_baseline = newdata.shape[0]
    newdata = newdata.loc[newdata.index.repeat(len(predict_times))].reset_index(drop=True)
    newdata['followup_time'] = pd.Series(predict_times).repeat(n_baseline).reset_index(drop=True)

    return newdata


def assert_matrix(mat, mode="numeric"):
    if not isinstance(mat, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if mode == "numeric" and not np.issubdtype(mat.dtype, np.number):
        raise ValueError("Matrix must be numeric.")

def assert_monotonic(arr, increasing=True):
    if increasing:
        if not np.all(np.diff(arr) >= 0):
            raise ValueError("Array is not monotonically increasing.")
    else:
        if not np.all(np.diff(arr) <= 0):
            raise ValueError("Array is not monotonically decreasing.")

def calculate_cum_inc(p_mat):
    assert_matrix(p_mat, mode="numeric")
    result = 1 - calculate_survival(p_mat)
    assert_monotonic(result)
    return result

def calculate_survival(p_mat):
    assert_matrix(p_mat, mode="numeric")
    cumprod_matrix = np.cumprod(1 - p_mat, axis=1)
    result = np.mean(cumprod_matrix, axis=0)
    assert_monotonic(result, increasing=False)
    return result

def calculate_predictions(newdata, model, treatment_values, pred_fun, coefs_mat, matrix_n_col):
    model_terms = model.feature_names_in_
    predictions = {}
    
    for treatment_label, treatment_value in treatment_values.items():
        newdata["assigned_treatment"] = treatment_value
        model_matrix = newdata[model_terms].values
        
        pred_list = [
            pred_fun(np.dot(model_matrix, coefs_mat[i, :].T).reshape(-1, matrix_n_col))
            for i in range(coefs_mat.shape[0])
        ]
        predictions[treatment_label] = np.column_stack(pred_list)
    
    return predictions

