# 1. Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 2. Load the dataset
url = "https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv"
insurance = pd.read_csv(url)

# 3. Preprocess categorical features
cat_cols = ['sex', 'smoker', 'region']
ct = ColumnTransformer([
    ("encoder", OneHotEncoder(sparse=False),
     [insurance.columns.get_loc(c) for c in cat_cols])
], remainder="passthrough")

data = ct.fit_transform(insurance)
# Get feature names (optional, for tracking)
feature_names = (ct.named_transformers_['encoder']
                 .get_feature_names_out(cat_cols).tolist() +
                 [c for c in insurance.columns if c not in cat_cols])

# 4. Split features and labels
X = pd.DataFrame(data, columns=feature_names)
y = insurance['expenses']

# 5. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train a simple linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 8. Evaluate the model
preds = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, preds)
print(f"Test Mean Absolute Error: {mae:.2f}")

if mae < 3500:
    print("✅ Challenge passed!")
else:
    print("❌ MAE should be below 3500, try improving it.")

# 9. Plot predictions vs true values
plt.figure(figsize=(6,6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([0, 60000], [0, 60000], color='red', linewidth=1)
plt.xlabel("True Expenses")
plt.ylabel("Predicted Expenses")
plt.title("True vs Predicted Healthcare Costs")
plt.show()
