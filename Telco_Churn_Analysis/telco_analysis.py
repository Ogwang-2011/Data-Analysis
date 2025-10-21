import pandas as pd

# Load Telco dataset
df = pd.read_csv(r"C:\Users\DELL\Documents\Telco_Customer_Churn_Dataset.csv")



# Preview
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Encode target variable
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop CustomerID (not useful for prediction)
df = df.drop("customerID", axis=1)

# Convert categorical columns
df = pd.get_dummies(df, drop_first=True)

# Split data
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Predict probabilities for all customers
df["Churn_Probability"] = model.predict_proba(X)[:,1]

# Save for Power BI
df_out = df.copy()
df_out.to_csv("Telco_with_Predictions.csv", index=False)
