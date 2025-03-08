from trial_sequence.utils import te_weights_fitted


def show_weight_models(object: object) -> None:
    if hasattr(object, "censor_weights") and object.censor_weights is not None:
        if test_list(object.censor_weights.fitted, te_weights_fitted):
            print("Weight Models for Informative Censoring")
            print("---------------------------------------\n")
            for name in object.censor_weights.fitted.keys():
                print(f"[[{name}]]")
                print(object.censor_weights.fitted[name])

    if hasattr(object, "switch_weights") and object.switch_weights is not None:
        if test_list(object.switch_weights.fitted, te_weights_fitted):
            print("Weight Models for Treatment Switching")
            print("-------------------------------------\n")
            for name in object.switch_weights.fitted.keys():
                print(f"[[{name}]]")
                print(object.switch_weights.fitted[name])


def test_list(object: dict, type: any) -> bool:
    if not isinstance(object, dict):
        return False

    return all([isinstance(obj, type) for obj in object.values()])
