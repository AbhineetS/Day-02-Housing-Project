# Day 2/64: House Price Prediction (Regression)
# We're predicting a continuous number (price), not a category (survived/died).
#
# To run this:
# 1. Activate your venv (find the 'activate' file in Day-01/venv/bin and run it)
# 2. Run: python3 run_house_price_model.py

print("--- [Day 2/64] Starting House Price Prediction Model ---")

# --- Step 1: Import Libraries ---
try:
    import pandas as pd
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    print("[Step 1/6] All libraries imported successfully.")
except ImportError:
    print("Error: Libraries not found. Make sure 'pandas' and 'scikit-learn' are installed in your venv.")
    exit()

# --- Step 2: Load The Data ---
# No CSV needed! The data is built-in to scikit-learn.
housing = fetch_california_housing()

# Convert it to a Pandas DataFrame for easier use
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['MedHouseVal'] = housing.target # MedHouseVal is our target (Median House Value)

print("[Step 2/6] California Housing dataset loaded successfully.")
print(f"Data has {data.shape[0]} rows and {data.shape[1]} columns.")
print("--- First 5 Rows of Data ---")
print(data.head()) # Show first 5 rows

# --- Step 3: Define Features (X) and Target (y) ---
# We'll use all features to predict the median house value.
X = data.drop('MedHouseVal', axis=1) # X is all columns EXCEPT the price
y = data['MedHouseVal']              # y is ONLY the price

print("\n[Step 3/6] Features (X) and Target (y) defined.")

# --- Step 4: Split Data ---
# We do the same 80/20 split as Day 1.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("[Step 4/6] Data split into 80% training and 20% validation.")

# --- Step 5: Create and Train the Model ---
# We're using a new model: LinearRegression.
model = LinearRegression()

# Train the model on our training data
model.fit(X_train, y_train)
print("[Step 5/6] Model trained successfully (Linear Regression).")

# --- Step 6: Evaluate the Model ---
# We can't use "accuracy" for regression. We use "Mean Absolute Error".
# This tells us (on average) how "wrong" our price predictions were.
predictions = model.predict(X_val)
mae = mean_absolute_error(y_val, predictions)

print("\n--- Model Evaluation ---")
# The target value is in "hundreds of thousands of dollars" (e.g., 2.5 = $250,000)
# So, an MAE of ~0.53 means our model is, on average, off by ~$53,000.
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print("This means our model's price predictions are, on average,")
print(f"off by about ${mae * 100000:,.2f}.")

print("\n===================================")
print(f"🎉 Day 2/64 Complete! 🎉")
print("===================================\n")