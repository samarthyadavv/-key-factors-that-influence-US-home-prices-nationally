# First cell - imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Second cell - load data
data = pd.read_csv('processed_data.csv', index_col=0)
print("Data shape:", data.shape)
data.head()

# Third cell - prepare features and target
feature_cols = [col for col in data.columns if col.endswith('_YOY')]
X = data[feature_cols]
y = data['CSUSHPISA']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fourth cell - scale features and train model
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Fifth cell - evaluate model
# Calculate performance
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R-squared Score: {r2:.4f}")
print(f"Root Mean Square Error: {rmse:.4f}")

# Sixth cell - visualize feature importance
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': abs(model.coef_)
})
importance = importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance, x='Importance', y='Feature')
plt.title('Feature Importance in Home Price Prediction')
plt.tight_layout()
plt.show()