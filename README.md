# Day 2/64: House Price Predictor (Linear Regression)

This is the second project in my 64-Day AI/ML Challenge.

## Objective
To build a **Regression** model, which predicts a continuous value (a price) rather than a category (like "survived/died").

## Workflow
1.  **Load Data**: Used the built-in `fetch_california_housing` dataset from scikit-learn.
2.  **Define Features**: Separated the data into features (e.g., house age, location) and the target (median house value).
3.  **Train Model**: Built and trained a `LinearRegression` model on 80% of the data.
4.  **Evaluate**: Tested the model on the remaining 20% and measured its performance using **Mean Absolute Error (MAE)**.

## Result
* **Mean Absolute Error (MAE): 0.5332**
* This means, on average, the model's price predictions are off by **$53,320.01**. This is a great baseline result for a simple linear model!

### How to Run This
1.  Ensure you have `scikit-learn` and `pandas` installed in your Python environment.
2.  Run the script: `python3 run_house_price_model.py`
