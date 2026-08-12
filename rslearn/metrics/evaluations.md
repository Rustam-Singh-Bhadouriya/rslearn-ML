# 📊 rslearn.metrics.evaluations

## Overview

The `evaluate_model()` function is responsible for evaluating machine learning models. It provides functions to evaluate the model's predictions against the true values and generate various evaluation metrics.

## Parameter


### `evaluate_model(model=None, X=None, y_pred=None, y_true=None, task="regression")`

- **Description**: Evaluates a machine learning model using input data (`X` and `y`) and compares it with the actual labels (`y_true`). The function supports both regression and classification tasks.
- **Parameters**:
  - `model`: A trained machine learning model that can be used to make predictions on the data.
  - `X`: Input features as a NumPy array or pandas DataFrame.
  - `y_pred`: Predicted target values for the input features.
  - `y_true`: True target values for the input features.
  - `task`: A string indicating whether the task is regression (`"regression"` or `"classification"`).


## Usage

1. **Import the `evaluate_model` module**:
   ```python
   from rslearn.metrics import evaluate_model
   ```

2. **Train a machine learning model and get predictions**:
   ```python
   # Assuming you have already trained your model using some training data
   model = LinearRegression()  # Example of training a linear regression model
   X_train, y_train = load_data('train.csv')
   model.fit(X_train, y_train)

   # Make predictions on the test data
   X_test, y_test = load_data('test.csv')
   y_pred = model.predict(X_test)
   ```

3. **Evaluate the model**:
   ```python
   # Evaluate the model using regression task
   evaluation = evaluate_model(model=model, X=X_test, y_true=y_test, task="regression")
   print(f"Regression Evaluation: {evaluation}")

   # Evaluate the model using classification task
   y_pred_classified = (y_pred > 0.5).astype(int)  # Assuming a threshold of 0.5 for classification
   evaluation_classified = evaluate_model(model=model, X=X_test, y_true=y_test_classified, task="classification")
   print(f"Classification Evaluation: {evaluation_classified}")
   ```

## NOTE
1. You can Give `y_pred` & `y_true` to get evaluation but they must be `np.array` or `pd.DataFrame` or Python Built in Data Stucture without any `Tensors` and they should be 1D array.
2. You can Give trained `Model` & `X` to get evaluation also but Model should not be trained with `TensorFlow`, `Keras` or `PyTorch`.
3. To use `TensorFlow` or `Keras` or `PyTorch` Model do this: 
    ``` python
    predictions = model.predict(X_test) # Most Likely It will return Tensor if trained.
    predictions = predictions.numpy() # Or According to FrameWork
    predictions = predictions.reshape(1, -1)

    evaluations = evaluate_model(y_pred=predictions, y_true=y_test, task="YOUR TASK eg regression or classification")
    ```