# 🧠 rslearn — Machine Learning Library (From Scratch)

**rslearn** is a lightweight, from-scratch machine learning library inspired by scikit-learn, built using pure Python and NumPy.

This project is focused on deeply understanding ML algorithms by implementing them step-by-step, while also providing a clean and usable API similar to modern ML libraries.

---

## Release & Changes
* **Version : ____**
* **Release Date: _______**

## 🚀 Features

### Latest (In metrics/_classification.py): 
* Confusion Metrics
* Recall
* Precision
* F1 Score

## 🐛 Fixes
* Module Base Design
* Simplicity In Code Style

## 🗄️ New File
* **_base.py**

### 📊 Linear Models

* Linear Regression (Single & Multi-feature)
* Logistic Regression (Binary & Multi-class)
* Ridge Regression (L2 Regularization)
* Lasso Regression (L1 Regularization)
* Elastic Net (L1 + L2)

---

### 📏 Metrics

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score
* Accuracy (for classification)

✔ Supports **single-output and multi-output** tasks

---

### 🔧 Preprocessing

* StandardScaler
* MinMaxScaler

---

### 🧪 Model Selection

* Train-Test Split

  * Supports `stratify` for balanced sampling

---

## ⚙️ Optimization Details

All models in **rslearn** are implemented using **Gradient Descent**.

⚠️ **Important:**

* Feature scaling is highly recommended for stable and faster convergence.
* Use:

  * `StandardScaler` (recommended)
  * or `MinMaxScaler`

---

## 🤖 Auto Scaling (Ridge, Lasso, ElasticNet)

Regularized models include:

```python
scale=True  # default
```

* Automatically applies feature scaling internally
* Helps prevent numerical instability

💡 Still recommended:

> Use `StandardScaler` manually for best performance and control.

---

## 📁 Project Structure

```
rslearn/
│
├── linear_model/
│   ├── _linear_regression.py
│   ├── _logistic_regression.py
│   ├── _ridge.py
│   ├── _lasso.py
│   ├── _elastic_net.py
│
├── preprocessing/
│   ├── _scaler.py
│
├── metrics/
│   ├── _regression.py
│
├── model_selection/
│   ├── _split.py
│
└── README.md
```

📌 Each module contains its own **detailed README** with usage examples and explanations.

---

## 🛠️ Installation

### Clone the repository

```bash
git clone https://github.com/Rustam-Singh-Bhadouriya/sklearn_clone.git
cd rslearn
```

### Install Usable Library
``` bash
pip install rslearn-ML
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📌 Quick Example

```python
from rslearn.linear_model import LinearRegression
from rslearn.preprocessing import StandardScaler
import numpy as np

X = np.array([10, 20, 30])
y = np.array([5, 10, 15])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

print(model.predict([40]))
```

---

## 📚 Documentation

* Each folder includes its own **README.md**
* Covers:

  * Usage
  * Parameters
  * Examples
  * Internal working

---

## 🎯 Goals of this Project

* Understand ML algorithms from scratch
* Build a sklearn-like API
* Create reusable and modular ML components
* Learn real-world ML system design

---

## 🧑‍💻 Author

**Rustam Singh Bhadouriya**

---

## 📜 License

This project is licensed under the MIT License.
