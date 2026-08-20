# linear_model.Ridge, Lasso, ElasticNet

## Overview
Ridge (l2), Lasso (l1) and ElasticNet (l1+l2) are algorithms to control over overfitting they uses LinearRegression Internaly with `regulization` parameter  
**Features**  
* Auto Scaling `scale=True` in `fit()`.
* Inbuilt Evaluations support.
* controll over scaling using `hard_scale_off=True`.
* Model saving in `.rslr` format.

## Parameters
Parameters are same in each Regulization Class.

### Class Parameter (tuning parameters)
| Parameters | Usage |
|---|---|
| alpha | Regularization strength (default: 0.1) |
| l1_ratio | L1 ratio for ElasticNet (default: 0.5) |
| max_itr | Maximum iterations (default: 3000) |
| min_loss | Minimum loss threshold (default: 0.1) |
| hard_scale_off | Disable scaling (default: False) |


### Method Parameters
* `fit(X, y, scale=True)`: Train the model
* `predict(X)`: Make predictions
* `evaluate(X=None, y_pred=None, y_true=None)`: Calculate metrics
* `save(file_name)`: Save model

## Usage

### Lasso (L1 Regularization)
```python
from rslearn.linear_model import Lasso
model = Lasso(alpha=0.1)
model.fit(X_train, y_train, scale=True)
```

### Ridge (L2 Regularization)
```python
from rslearn.linear_model import Ridge
model = Ridge(alpha=0.1)
model.fit(X_train, y_train, scale=True)
```

### ElasticNet (L1 + L2 Regularization)
```python
from rslearn.linear_model import ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train, scale=True)
```

### Each Contains Saving Capability but saves as LinearRegression
```python
model.save("regu_model.rsl") # Will be saved as .rslr
```

# NOTE
Using LinearRegression is Better than Using Law Regulization Class 'cause It provides more control over parameters.
```python
from rslearn.linear_model import LinearRegression

model = LinearRegression(regulization="l1") # Choose according to Table
```

| Regulization | Parameter Name |
|---|---|
| l1 | Lasso |
| l2 | Ridge |
| elastic_net | ElasticNet |


# Follow up
* [LogisticRegression](logistic_regression.md)
* [LinearRegression](regulizations.md)
* [train_test_split](../model_selection/splitting.md) To split Data to X_train, X_test, y_train, y_test
* [Model loading](../loader/load_model.md) To load saved Model