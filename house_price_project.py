import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample housing data
data = {
    "sqft": [750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400],
    "bedrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    "bathrooms": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
    "price": [150000, 160000, 165000, 180000, 195000, 200000, 220000, 250000, 270000, 290000]
}
df = pd.DataFrame(data)
print(df)

print("\n--- Describe ---")
print(df.describe())

print("\n--- Correlation ---")
print(df.corr())

# Features and label
X = df[["sqft", "bedrooms", "bathrooms"]]
y = df["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Model Results ---")
print("Weights:", model.coef_)
print("Bias:", model.intercept_)

# Predict and evaluate
predictions = model.predict(X_test)
print("\nPredictions:", predictions)
print("Actual:", y_test.values)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("\nMSE:", mse)
print("R² score:", r2)