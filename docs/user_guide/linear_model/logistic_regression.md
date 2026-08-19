# linear_model.LogisticRegression

## Overview
LogisticRegression is a classification task algorithm.  
**Features**  
* Auto Scaling `scale=True` in `fit()`.
* Inbuilt Evaluations support.
* controll over scaling using `hard_scale_off=True`.
* Model saving in `.rslc` format.
* Multi-Class Classiciation support with `auto`, `liblinear` & `saga` solver.

## Parameters

### Class Parameter (tuning parameters)

| Parameter | Description |
| --- | --- |
| solver | "liblinear" (binary), "saga" (categorical), "auto" (default) |
| lr | Learning rate (default: 0.001) |
| weights | Initial weights (optional) |
| bias | Initial bias (optional) |
| catogirical_model | List of models for categorical (internal use) |
| max_itr | Maximum iterations (default: 3000) |
| hard_scale_off | Disable scaling (default: False) |


**Usage**  
```python
from rslearn.linear_model import LogisticRegression

Model = LogisticRegression(solver="auto", lr=0.02, max_itr=18000, hard_scale_off=True)
```

## NOTE
`hard_scale_off=True` is not recommanded untill you used any other Scaler on the Data.

### Method Parameters
* `fit(X, y, scale=True)`: Train the model
* `predict(X)`: Make predictions
* `evaluate(X=None, y_pred=None, y_true=None)`: Calculate metrics
* `get_weight_bias()`: Get model parameters
* `save(file_name)`: Save model

## Usage
```python
from rslearn.linear_model import LogisticRegression

model = LogisticRegression(solver="auto", lr=0.02, max_itr=18000, hard_scale_off=False) # Set it to True if you already scaled the data

model.fit(X_train, y_train) # Fitting the Model

# Getting Evaluations with predictions at once
evaluations = model.evaluate(model=model, y_true=y_test, X=X_test)
preds = evaluations['predictions']

print(f"Predictions {preds}")
print(evaluations)

evaluations.save("my_model.rsl") # But will be saved as .rslc in LogisticRegression

```

# Follow up
* [LinearRegression](linear_regression.md)
* [Regulizatoins](regulizations.md)
* [train_test_split](../model_selection/splitting.md) To split Data to X_train, X_test, y_train, y_test
* [Model loading](../loader/load_model.md) To load saved Model