import pandas as pd


# In the actual `TrialEmulation`` package from R, they used an external C++ function to `censor_func`.
def censor_func(sw_data: pd.DataFrame) -> pd.DataFrame:
    n = len(sw_data)

    started0 = sw_data["started0"].copy()
    started1 = sw_data["started1"].copy()
    stop0 = sw_data["stop0"].copy()
    stop1 = sw_data["stop1"].copy()
    eligible0_sw = sw_data["eligible0_sw"].copy()
    eligible1_sw = sw_data["eligible1_sw"].copy()
    delete_ = sw_data["delete"].astype(bool).copy()

    t_first = sw_data["first"]
    t_eligible = sw_data["eligible"]
    t_treatment = sw_data["treatment"]
    t_switch = sw_data["switch"]

    started0_ = 0
    started1_ = 0
    stop0_ = 0
    stop1_ = 0
    eligible0_sw_ = 0
    eligible1_sw_ = 0

    for i in range(n):
        if t_first[i]:
            started0_ = 0
            started1_ = 0
            stop0_ = 0
            stop1_ = 0
            eligible0_sw_ = 0
            eligible1_sw_ = 0

        if stop0_ == 1 or stop1_ == 1:
            started0_ = 0
            started1_ = 0
            stop0_ = 0
            stop1_ = 0
            eligible0_sw_ = 0
            eligible1_sw_ = 0

        if started0_ == 0 and started1_ == 0 and t_eligible[i] == 1:
            if t_treatment[i] == 0:
                started0_ = 1
            elif t_treatment[i] == 1:
                started1_ = 1

        if started0_ == 1 and stop0_ == 0:
            eligible0_sw_ = 1
            eligible1_sw_ = 0
        elif started1_ == 1 and stop1_ == 0:
            eligible0_sw_ = 0
            eligible1_sw_ = 1
        else:
            eligible0_sw_ = 0
            eligible1_sw_ = 0

        if t_switch[i] == 1:
            if t_eligible[i] == 1:
                if t_treatment[i] == 1:
                    started1_ = 1
                    stop1_ = 0
                    started0_ = 0
                    stop0_ = 0
                    eligible1_sw_ = 1
                elif t_treatment[i] == 0:
                    started0_ = 1
                    stop0_ = 0
                    started1_ = 0
                    stop1_ = 0
                    eligible0_sw_ = 1
            else:
                stop0_ = started0_
                stop1_ = started1_

        if eligible0_sw_ == 0 and eligible1_sw_ == 0:
            delete_[i] = True
        else:
            started0[i] = started0_
            started1[i] = started1_
            stop0[i] = stop0_
            stop1[i] = stop1_
            eligible1_sw[i] = eligible1_sw_
            eligible0_sw[i] = eligible0_sw_
            delete_[i] = False

    sw_data["started0"] = started0
    sw_data["started1"] = started1
    sw_data["stop0"] = stop0
    sw_data["stop1"] = stop1
    sw_data["eligible0_sw"] = eligible0_sw
    sw_data["eligible1_sw"] = eligible1_sw
    sw_data["delete"] = delete_

    return sw_data
