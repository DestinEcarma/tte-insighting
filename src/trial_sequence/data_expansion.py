
import numpy as np
import pandas as pd

def expand(sw_data: pd.DataFrame, outcomeCov_var: list, where_var: list, use_censor: bool, maxperiod: int, minperiod: int, keeplist: list):
    temp_data = pd.DataFrame({
        "id": sw_data["id"],
        "period": sw_data["period"],
        "switch": sw_data["switch"],
        "wtprod": 1.0,
        "elgcount": 0.0,
        "expand": 0.0,
        "treat": 0.0,
        "dosesum": 0.0,
    })
    
    temp_data.loc[
        (sw_data["eligible"] == 1) & sw_data["treatment"].notna() &
        (minperiod <= sw_data["period"]) & (sw_data["period"] <= maxperiod), "expand"
    ] = 1
    
    sw_data.loc[sw_data["first"] == True, "weight0"] = 1.0
    sw_data["weight0"] = sw_data.groupby("id")["wt"].cumprod()
    
    temp_data["wtprod"] = sw_data["weight0"]
    temp_data["treat"] = sw_data["treatment"]
    temp_data["dosesum"] = sw_data["cumA"]
    temp_data["elgcount"] = sw_data["eligible"]
    temp_data.loc[sw_data["eligible"] == 1, "init"] = sw_data.loc[sw_data["eligible"] == 1, "treatment"]
    temp_data["init_shift"] = sw_data["treatment"].shift()
    temp_data.loc[sw_data["eligible"] == 0, "init"] = temp_data["init_shift"]
    temp_data.drop(columns=["init_shift"], inplace=True)
    
    if outcomeCov_var:
        temp_data[outcomeCov_var] = sw_data[outcomeCov_var]
    if where_var:
        temp_data[where_var] = sw_data[where_var]

    expand_index = np.repeat(np.arange(len(sw_data)), sw_data['period'] + 1)

    switch_data = pd.DataFrame({
        "id": sw_data.iloc[expand_index]["id"],
        "period_new": sw_data.iloc[expand_index]["period"],
        "cumA_new": sw_data.iloc[expand_index]["cumA"],
        "treatment_new": sw_data.iloc[expand_index]["treatment"],
        "switch_new": sw_data.iloc[expand_index]["switch"] if use_censor else 0,
        "outcome_new": sw_data.iloc[expand_index]["outcome"],
        "time_of_event": sw_data.iloc[expand_index]["time_of_event"],
        "weight0": sw_data.iloc[expand_index]["weight0"],
    })
    switch_data["trial_period"] = np.concatenate([np.arange(p + 1) for p in sw_data['period']])
    switch_data["index"] = np.arange(1, len(switch_data) + 1)
    
    switch_data = switch_data.merge(temp_data, how="left", left_on=["id", "trial_period"], right_on=["id", "period"])
    switch_data.sort_values(by="index", inplace=True)
    switch_data["followup_time"] = switch_data["period_new"] - switch_data["trial_period"]
    
    if "dose" in keeplist:
        switch_data["dose"] = switch_data["cumA_new"] - switch_data["dosesum"] + switch_data["treat"]

    switch_data.loc[switch_data["followup_time"] == 0, "switch_new"] = 0

    for (_, _), group in switch_data[switch_data['expand'] == 1].groupby(['id', 'trial_period']):
        switch_data.loc[group.index, 'expand'] = expand_until_switch(group['switch_new'])

    switch_data["weight"] = (switch_data["weight0"] / switch_data["wtprod"]).dropna(axis=0)
    switch_data["case"] = 0
    
    if not use_censor:
        switch_data.loc[(switch_data["time_of_event"] == switch_data["period_new"]) & (switch_data["outcome_new"] == 1), "case"] = 1
    else:
        switch_data.loc[switch_data["switch_new"] == 1, "case"] = np.nan
        switch_data.loc[(switch_data["switch_new"] == 0) & (switch_data["time_of_event"] == switch_data["period_new"]) & (switch_data["outcome_new"] == 1), "case"] = 1
    
    switch_data.rename(columns={
        "case": "outcome",
        "init": "assigned_treatment",
        "treatment_new": "treatment"
    }, inplace=True)
    
    switch_data = switch_data[switch_data["expand"] == 1]
    return switch_data[keeplist]

def expand_until_switch(s):
    first_switch = np.argmax(s == 1) if np.any(s == 1) else -1

    if first_switch != -1:
        return np.concatenate([np.ones(first_switch), np.zeros(len(s) - first_switch)])

    return np.ones(len(s))