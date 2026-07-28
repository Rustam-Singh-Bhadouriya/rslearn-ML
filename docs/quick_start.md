# Quick Start Guide: rslearn-ML

# Objective: Get Started with rslearn-ML

## NOTE
This is practicle & code only file for Information in the depth of parameter and how it works consider user guide

* <a href="user_guide/README.md">user guide (full)</a>

### Installation 
install the library:
``` bash
pip install --upgrade rslearn-py
```

### Import Necessary Modules
Start by Importing LinearRegression & Numpy:
``` py
from rslearn.linear_model import LinearRegression
import numpy as np
```

### Check version
Check library github and pypi version:
```py
import rslearn
print(rslearn.__version__) # pypi
print(rslearn.__github_version__) # github version
```

### Load Sample Data
Loading Dataset easy version 10k rows:
``` py
import kagglehub as kh
import pandas as pd

path = kh.dataset_download("rustambhadouriya/synthetic-housing-price-dataset")

df = pd.read_csv(f"{path}/housing_dataset.csv")

X = df.drop(columns=["price"])
y = df.price
```

### Create and Configure the Model
Initialize the model with desired parameters:
``` py
model = LinearRegression(regulization="l1", max_itr=4000) # or hard_scale_off=True
```

### Split Data and Fit the Model
Split into training and testing sets, then train the model:
``` py
from rslearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
```

### Make Predictions
Use the trained model to predict outcomes:
``` py
pred = model.predict(X_test)
```

### Make Evaluations
Use Inbuild Evaluation to Evaluate All Metrics Algorithams at once with 1 line of code: 
``` py
evals = model.evaluate(y_pred=pred, y_true=t_test)
print(evals)
```

or you can just give `X_test` also.

``` py
evals = model.evaluate(X=X_test, y_true=y_test)
print(evals)
```

# For a clear guide visit user guide