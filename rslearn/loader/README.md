# Model Loader Documentation

## Overview
The `rslearn/loader/_model_loader.py` module provides functions to load pre-trained models saved in `.rslr`, and `.rslc` formats. These models include Logistic Regression and Linear Regression, each with various configurations for their solvers, regularization, and scaling.

## Functions

### load_model(file_path: str = "rslearn_model.rslr/c")

**Overview**
This function is the main entry point to load a pre-trained model based on the file extension. It supports loading both `.rslr` (Linear Regression) and `.rslc` (Logistic Regression) models.

**Parameters**
- `file_path`: The path to the saved model file. Default is "rslearn_model.rsl".

**Returns**
- A pre-trained model instance.

**Raises**
- `Error`: If the file extension is not supported.
- `Error`: If the file is invalid or corrupted.

## Model Structure

### LinearRegression and LogisticRegression

Both models have a common structure with various parameters:

- `regulization` (LinearRegression only)
- `alpha`
- `l1_ratio` (LinearRegression only)
- `min_loss`
- `lr` (Learning rate)
- `weights`
- `bias`
- `hard_scale_off` (Flag to disable scaling)
- `max_itr` (Maximum iterations)

### Scaling and Fitting

Models also handle data scaling. If the model was trained with scaling, it will preserve this information when loaded:

- `Scaler.mean` and `Scaler.std` (if primary scaled)
- `backup_scaler.maxx` (if not primary scaled)

## Example Usage

```python
from rslearn.loader import load_model

# Load a Linear Regression model
model_lr = load_model("path/to/model.rslr")

# Load a Logistic Regression model
model_logistic = load_model("path/to/model.rslc")
```