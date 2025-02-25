import numpy as np
import pandas as pd

from trial_sequence.censor_func import censor_func


def data_manipulation(data: pd.DataFrame, use_censor=True):
    assert isinstance(use_censor, bool)

    data["after_eligibility"] = data.groupby("id")["period"].transform(
        lambda x: (
            x >= x[data["eligible"] == 1].min()
            if (data["eligible"] == 1).any()
            else np.inf
        )
    )
    data = data[data["after_eligibility"]]
    data.drop(columns=["after_eligibility"], inplace=True)

    data["after_event"] = data.groupby("id")["period"].transform(
        lambda x: (
            x > x[data["outcome"] == 1].min()
            if (data["outcome"] == 1).any()
            else np.inf
        )
    )
    data = data[~data["after_event"]]
    data.drop(columns=["after_event"], inplace=True)

    event_data = (
        data.groupby("id").last().reset_index()[["id", "period", "outcome"]]
    )
    event_data["time_of_event"] = np.where(
        event_data["outcome"] == 1, event_data["period"].astype(float), 9999
    )
    event_data = event_data[["id", "time_of_event"]]

    sw_data = data.merge(event_data, on="id", how="left")
    sw_data["first"] = ~sw_data["id"].duplicated()
    sw_data["am_1"] = sw_data.groupby("id")["treatment"].shift()

    sw_data["cumA"] = np.where(sw_data["first"], True, np.nan)
    sw_data["am_1"] = np.where(sw_data["first"], 0, sw_data["am_1"])
    sw_data["switch"] = np.where(sw_data["first"], 0, 1)
    sw_data["regime_start"] = np.where(
        sw_data["first"], sw_data["period"], sw_data["period"]
    )
    sw_data["time_on_regime"] = np.where(sw_data["first"], 0, 0)

    sw_data.loc[
        ~sw_data["first"] & (sw_data["am_1"] != sw_data["treatment"]), "switch"
    ] = 1
    sw_data.loc[
        ~sw_data["first"] & (sw_data["am_1"] == sw_data["treatment"]), "switch"
    ] = 0

    sw_data.loc[
        ~sw_data["first"] & (sw_data["switch"] == 1), "regime_start"
    ] = sw_data["period"]
    sw_data["regime_start"] = sw_data.groupby("id")["regime_start"].ffill()

    sw_data["regime_start_shift"] = sw_data.groupby("id")[
        "regime_start"
    ].shift()
    sw_data.loc[~sw_data["first"], "time_on_regime"] = sw_data[
        "period"
    ] - sw_data["regime_start_shift"].astype(float)

    sw_data.loc[sw_data["first"], "cumA"] += sw_data["treatment"]
    sw_data.loc[~sw_data["first"], "cumA"] = sw_data["treatment"]
    sw_data["cumA"] = sw_data.groupby("id")["cumA"].cumsum()
    sw_data.drop(columns=["regime_start_shift"], inplace=True)

    if use_censor:
        for col in [
            "started0",
            "started1",
            "stop0",
            "stop1",
            "eligible0_sw",
            "eligible1_sw",
            "delete",
        ]:
            sw_data[col] = np.nan
        sw_data = censor_func(sw_data)
        sw_data = sw_data[sw_data["delete"] == False]
        sw_data.drop(
            columns=[
                "delete",
                "eligible0_sw",
                "eligible1_sw",
                "started0",
                "started1",
                "stop0",
                "stop1",
            ],
            inplace=True,
        )

    sw_data["eligible0"] = (sw_data["am_1"] == 0).astype(int)
    sw_data["eligible1"] = (sw_data["am_1"] == 1).astype(int)

    sw_data.sort_values(by="id", inplace=True)
    return sw_data
