import numpy as np


def check_multioutput(parameter) -> None:
    valid_params = {
        "weighted",
        "uniform_average",
        "raw_values"
        }

    if parameter not in valid_params:
        raise ValueError(
            f"Got Invalid Parameter, {parameter}. But supported {valid_params} only."
            )

    return

def convert_array(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return y_true, y_pred

def convert1D(*args) -> np.array:
    """
    Parameters
    ----------
    y_true,  
    y_pred,  
    any, any  

    Returns
    -------
    list of Flat Arrays by  
    X, y = convert1D(X, y)   
    X, y, z = convert1D(X, y, z)  

    """
    arrays = []
    for items in args:
        items, _ = convert_array(items, [1012, 1203])
        arrays.append(np.ravel(items))
    
    return tuple(np.array(arrays))

def shape_checker(): ...;

if __name__ == "__main__":
    X = [[12], [11], [10]]
    y = [[13], [11], [8]]
    X, y= convert1D(X, y)
    print(type(X), y)

