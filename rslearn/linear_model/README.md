# Linear Models Documentation

## Changes Made
1. Parameter Moved to `__init__()` from `fit()`  
2. Better Scaling Optimization when `scale=False`  
3. new parameter `hard_scale_off` to disable all scalers on data  
4. Model.save() in All `linear_model` family.  
5. Custom format `.rslr`, `.rslc`  
---

|extension|usage|  
|---------|--------|  
| `.rslr` | saving format for `LinearRegression` & `Regulizations`|  
| `.rslc` | saving format for `LogisticRegression`|  
---

## Linear Regression

### Overview
Linear Regression implementation using gradient descent with optional regularization (L1, L2, ElasticNet).

### Parameters
- `regulization`: Regularization type ("l1", "l2", "elastic_net", None)
- `alpha`: Regularization strength (default: 0.1)
- `l1_ratio`: L1 ratio for ElasticNet (default: 0.5)
- `lr`: Learning rate (default: 0.001)
- `max_itr`: Maximum iterations (default: 3000)
- `weights`: Initial weights (optional)
- `bias`: Initial bias (optional)
- `min_loss`: Minimum loss threshold (default: 0.1)
- `hard_scale_off`: Disable scaling (default: False)

### Methods
- `fit(X, y, scale=True)`: Train the model
- `predict(X)`: Make predictions
- `evaluate(X=None, y_pred=None, y_true=None)`: Calculate metrics
- `get_weight_bias()`: Get model parameters
- `save(file_name)`: Save model

### Example
```python
from rslearn.linear_model import LinearRegression
model = LinearRegression(regulization="l1")
model.fit(X_train, y_train, scale=True)
predictions = model.predict(X_test)
metrics = model.evaluate(y_pred=predictions, y_true=y_test)
model.save("linear.rsl") # will be saved as linear.rslr
```

## Logistic Regression

### Overview
Logistic Regression for binary and categorical classification with automatic solver selection.

### Parameters
- `solver`: "liblinear" (binary), "saga" (categorical), "auto" (default)
- `lr`: Learning rate (default: 0.001)
- `weights`: Initial weights (optional)
- `bias`: Initial bias (optional)
- `catogirical_model`: List of models for categorical (internal use)
- `max_itr`: Maximum iterations (default: 3000)
- `hard_scale_off`: Disable scaling (default: False)

### Methods
- `fit(X, y, scale=True)`: Train the model
- `predict(X)`: Make predictions
- `predict_proba(X)`: Get class probabilities
- `evaluate(X=None, y_pred=None, y_true=None)`: Calculate metrics
- `save(file_name)`: Save model

### Example
```python
from rslearn.linear_model import LogisticRegression
model = LogisticRegression(solver="auto")
model.fit(X_train, y_train, scale=True)
predictions = model.predict(X_test)
metrics = model.evaluate(y_pred=predictions, y_true=y_test)
model.save("logistic.rsl") # will be saved as logistic.rslc
```

## Regularization Models

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

### Common Parameters for Regularization Models
- `alpha`: Regularization strength (default: 0.1)
- `l1_ratio`: L1 ratio for ElasticNet (default: 0.5)
- `min_loss`: Minimum loss threshold (default: 0.1)
- `max_itr`: Maximum iterations (default: 3000)
- `hard_scale_off`: Disable scaling (default: False)

### Methods
- `fit(X, y, scale=True)`: Train the model
- `predict(X)`: Make predictions
- `evaluate(X=None, y_pred=None, y_true=None)`: Calculate metrics
- `save(file_name)`: Save model

### For Beginner friendly guide visit docs/