# rslearn-ML
# Copyright (C) 2026 Rustam Singh Bhadouriya
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the LICENSE file for more details.

import numpy as np
from rslearn.Errors import *
from rslearn import BaseEstimators

def evals_params(model, X, y_pred, y_true, task):

    if task not in ("regression", "classification"):
        raise InvalidValueError(f"given task does not supported by function, {task}, supported task {("regression", "classification")}")
    if y_true is None:
        raise InvalidValueError(f"`y_true` is given `None`")
    if model is None:
        if y_pred is None:
            raise InvalidValueError("`y_pred` & `model` both are given `None`")
        else:
            pass # Nothing to do here
    else:
        if y_pred is not None:
            pass 
        else:
            # X is also None then Nothing to get pred even though model exits
            if X is None:
                raise InvalidValueError("`X` & `y_pred` both are given `None`")
            else:
                # if X exits
                y_pred = model.predict(X)

    evaluator = BaseEstimators.BaseEstimator()
    # configs
    evaluator._fitted = True
    evaluator.type = task



    evaluations = evaluator._eval(y_pred=y_pred, y_true=y_true)
    return evaluations
                
        
        

def evaluate_model(model=None, X=None, y_pred=None, y_true=None, task="regression"):
    """
    Evaluate the Model with one line of code.  
    NOTE: This Function Does not support ``PyTorch``,``TensorFlow`` or ``keras`` Model and neither ``Tensors`` To use this Function get prediction from model and convert them to 1D ``np.array``

    Parameters
    ----------
    model: default=None, `linear_model`, `neighbors`  
        Model to get predictions on ``X``  
    X: default=None, `np.array`, `pd.DataFrame`  
        NxM metrics to get predictions if ``y_pred=None``  
    y_pred: default=None, `np.array`  
        prediction from Model  
    y_true: default=None, `np.array`, `pd.DataFrame`  
        True values to evaluate, should not be ``None``
    task: default="regression", str, `classification`  
        Task selection for selective metrics  
    
    NOTE: This Function works with ``rslearn``, ``scikit-learn``, ``xgboost`` etc.  
    NOTE: This Function does not support probabilities (classification)/raw values from any Model.  
    NOTE: Select correct `task` type for correct evaluation  
    """
    return evals_params(model=model, X=X, y_pred=y_pred, y_true=y_true, task=task)

