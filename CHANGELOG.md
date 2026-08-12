# CHANGELOG 

All notable changes to this project will be documented in this file.

---

## [1.1.0 - v1.0.4] - 2026-MM-DD

### Feature/ref/fix/doc - 1.1.0
* Doc Update according new changes, README.md under folders  
* Doc String update
* added github version info, `rslearn.__github_version__`
* Implemented `evaluate_model()` info=[docs](rslearn/metrics/evaluations.md)

### Feature/ref/fix/doc - 1.0.9
* Model Saving among all `linear_model` family.
* Class refactor with BaseEstimator Classes (neighbors, linear_model)
* `hard_scale_off` special parameter
* parameters moved to `__init__()` from `fit()`
* Model loading
* custom format `.rslr`, `.rslc` - Read `linear_model/README.md` for more info
* pipeline bug for only StandardScaler usage fixed
* Improved StandardScaler & MinMaxScaler Security
* Improved BackupScaler
* Version info update, `rslearn.__version__`
* Open Basic Test Cases


### Tested - 1.0.9
* Tested on iris & California Hoursing Dataset
* Tested on Kaggle Dataset, 500k rows dataset
**For More Testing Info Visit Kaggle ``ItzRustam``**  

### Notes
***`See Older Version for more Information`***  
* for more about `docs/` read `docs/README.md`