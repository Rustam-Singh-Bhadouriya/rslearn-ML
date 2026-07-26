# Preprocessing

A lightweight and efficient preprocessing module inspired by scikit-learn, 
designed for simplicity, speed, and ease of use.

This module eliminates common friction points while working with feature scaling,
especially for beginners and rapid prototyping.

---

## 🚀 Features

- Minimal and clean API
- Handles both 1D and 2D data automatically
- Works with Python lists and NumPy arrays
- Lightweight alternative to large ML libraries

---

## 📦 Included Components

This module provides essential scalers to standardize your data, which is recommended for gradient-based algorithms like Linear Regression or Logistic Regression.

- `StandardScaler`: Computes the mean and standard deviation to normalize features to zero mean and unit variance.
    - **Best For**: Small datasets where feature distribution doesn't have extreme outliers (though robust in general).

- `MinMaxScaler`: Scales data by removing min/max values and scaling it to a specified range, typically $[0, 1]$.
    - **Best For**: Large datasets or when you need features constrained to a specific fixed range.

---

## ⚙️ How Scalers Work

Both scalers implement the following core methods:

### `fit(data)`
Computes and stores the necessary statistics (mean, standard deviation for `StandardScaler`, min/max values for `MinMaxScaler`) from the training data (`X`).

### `transform(data)`
Applies the learned transformation to new or existing data. This method ensures that the scaling applied during `fit` is correctly used here.

### `fit_transform(data)`
A convenience method that performs both fitting and transformation in one step, returning the fully scaled array.

---

## 🛠️ Installation / Usage

```python
from rslearn.preprocessing import StandardScaler, MinMaxScaler

# --- Using StandardScaler for standardization (Zero Mean, Unit Variance) ---
scaler_std = StandardScaler()
# Fit and transform your data
X_scaled_std = scaler_std.fit_transform(your_data_array) 

print("Standardized Data:\n", X_scaled_std)


# --- Using MinMaxScaler for scaling to a range (e.g., [0, 1]) ---
scaler_minmax = MinMaxScaler(feature_range=(0, 1)) # Scales data between 0 and 1
X_scaled_minmax = scaler_minmax.fit_transform(your_data_array)

print("MinMax Scaled Data:\n", X_scaled_minmax)


# --- Inverse Transformation (Reversing the scaling) ---
# To go back to original values, use inverse_transform on a fitted object:
# X_original = scaler_std.inverse_transform(X_scaled_std)
```