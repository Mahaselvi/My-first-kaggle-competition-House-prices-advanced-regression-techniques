#  House Price Prediction — Kaggle Competition

A machine learning pipeline built for the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) Kaggle competition. The goal is to predict residential home sale prices using 79 explanatory variables describing almost every aspect of the homes.

---

## Project Structure

```
house-price-prediction/
├── house_prediction.ipynb   # Main notebook
├── train.csv                # Training dataset (from Kaggle)
├── test.csv                 # Test dataset (from Kaggle)
└── submission.csv           # Final predictions for Kaggle submission
```

---

## Key Learnings

### 1. Log Transformation of the Target Column
Sale prices are right-skewed — a few very expensive homes can distort the model. Applying `np.log1p()` to `SalePrice` before training compresses this skew, making the distribution more normal. This helps the model learn general patterns rather than chasing outliers. Predictions are converted back using `np.expm1()` before the final submission.

```python
y = np.log1p(df['SalePrice'])
```

### 2. Splitting Categorical and Numerical Features
Different feature types need different preprocessing strategies. Numerical columns (dtype `int64`/`float64`) and categorical columns (dtype `object`) are identified and separated right at the start, so each group can be handled with the most suitable approach independently.

```python
num_cols = x.select_dtypes(include=['int64', 'float64']).columns
cat_cols = x.select_dtypes(include=['object']).columns
```

### 3. Handling Missing Values with SimpleImputer and Pipeline
- **Numerical features** use `SimpleImputer(strategy='mean')` to fill missing values with the column mean.
- **Categorical features** use a mini-pipeline: `SimpleImputer(strategy='most_frequent')` fills gaps with the most common category, followed by `OneHotEncoder` to convert text labels into numeric format. `handle_unknown='ignore'` ensures unseen categories in the test set don't cause errors.

```python
num_transformer = SimpleImputer(strategy='mean')

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
```

> **Note:** Some features like `PoolQC`, `Alley`, and `FireplaceQu` have `NaN` values that actually mean *"not present"* — not truly missing data. These are replaced with the string `'None'` before imputation so that the model treats them as a valid category.

### 4. ColumnTransformer for Feature Transformation
`ColumnTransformer` combines both transformers into a single, unified preprocessing step — applying the numerical strategy to numeric columns and the categorical strategy to categorical columns simultaneously.

```python
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])
```

### 5. End-to-End Model Pipeline
The preprocessor and the model (`RandomForestRegressor`) are wrapped into a single `Pipeline`. This ensures that preprocessing is always applied consistently during both training and prediction — no data leakage, no manual re-application of transforms.

```python
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42, n_jobs=-1))
])
```

### 6. Hyperparameter Tuning with GridSearchCV
`GridSearchCV` systematically tries all combinations of specified hyperparameters and selects the best model configuration based on R² score using 3-fold cross-validation.

```python
param_grid = {
    'model__n_estimators': [200, 300],
    'model__max_depth': [20, None],
    'model__max_features': [0.4, 0.5]
}
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)
```

### 7. Evaluation: R² Score and Root Mean Squared Log Error (RMSLE)
Two metrics are used to evaluate model performance:

- **R² Score** — measures how well the model explains variance in the validation set (closer to 1.0 is better).
- **RMSLE** — since predictions were made on log-transformed targets, the RMSE computed is effectively the Root Mean Squared Log Error. This is also the official Kaggle evaluation metric for this competition.

```python
print(f"R² on Validation Set: {r2_score(y_test, y_pred):.4f}")
print(f"Root Mean Squared Log Error: {root_mean_squared_error(y_test, y_pred):.4f}")
```

### 8. Generating the Submission File
After predicting on the test set, predictions are inverse-transformed from log scale back to dollar values using `np.expm1()`, then saved as a `submission.csv` ready for Kaggle upload.

```python
test_pred_dollars = np.expm1(best_model.predict(test_df))
submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_pred_dollars})
submission.to_csv("submission.csv", index=False)
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `scikit-learn` | Preprocessing, pipeline, model, and evaluation |
| `RandomForestRegressor` | Regression model |
| `GridSearchCV` | Hyperparameter tuning |
| `ColumnTransformer` | Unified feature preprocessing |
| `Pipeline` | End-to-end ML workflow |

---

## How to Run

1. Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).
2. Place them in the project root alongside the notebook.
3. Open `house_prediction.ipynb` and run all cells.
4. The `submission.csv` file will be generated in the project root.

---

## Competition

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) — Kaggle
