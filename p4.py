import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Cleaned_Healthcare_Analytics.csv")

# Display basic dataset information
print("First 5 rows:\n", df.head())
print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
df.info()
print("\nSummary Statistics:\n", df.describe())
print("\nColumn Names:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nUnique Values per Column:\n", df.nunique())

# Convert date columns to datetime format
df["Admission_Time"] = pd.to_datetime(df["Admission_Time"])
df["Discharge_Time"] = pd.to_datetime(df["Discharge_Time"])

# Extract hour from admission time for high-traffic analysis
df["Hour"] = df["Admission_Time"].dt.hour

# Step 3: EDA - Subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 15))

# Waiting Time Distribution
sns.histplot(df["Waiting_Time (min)"], bins=30, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Waiting Times")
axes[0, 0].set_xlabel("Waiting Time (minutes)")
axes[0, 0].set_ylabel("Frequency")

# High-Traffic Hours
sns.countplot(x=df["Hour"], palette="viridis", ax=axes[0, 1])
axes[0, 1].set_title("Patient Admissions by Hour")
axes[0, 1].set_xlabel("Hour of the Day")
axes[0, 1].set_ylabel("Number of Patients")

# Correlation Heatmap
corr = df[["Waiting_Time (min)", "Flu_Cases", "Hour"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=axes[1, 0])
axes[1, 0].set_title("Correlation Heatmap")

# Pie Chart of Severity Levels
severity_counts = df["Severity_Level"].value_counts()
axes[1, 1].pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%', colors=["red", "orange", "green"])
axes[1, 1].set_title("Distribution of Severity Levels")

# Box Plot for Waiting Times
sns.boxplot(x=df["Severity_Level"], y=df["Waiting_Time (min)"], palette="coolwarm", ax=axes[2, 0])
axes[2, 0].set_title("Box Plot of Waiting Times by Severity Level")

# Violin Plot for Flu Cases
sns.violinplot(x=df["Severity_Level"], y=df["Flu_Cases"], palette="muted", ax=axes[2, 1])
axes[2, 1].set_title("Violin Plot of Flu Cases by Severity Level")

plt.tight_layout()
plt.show()

# Swarm Plot for Waiting Times
plt.figure(figsize=(8, 5))
sns.swarmplot(x=df["Severity_Level"], y=df["Waiting_Time (min)"], palette="coolwarm")
plt.title("Swarm Plot of Waiting Times by Severity Level")
plt.show()

# Step 4: Predictive Analysis - Time Series Forecasting with Prophet
prophet_df = df[["Admission_Time", "Flu_Cases"]].rename(columns={"Admission_Time": "ds", "Flu_Cases": "y"})
model = Prophet()
model.fit(prophet_df)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
model.plot(forecast)
plt.show()

# Time Series Forecasting with ARIMA
flu_cases = df.set_index("Admission_Time")["Flu_Cases"].resample("D").mean().fillna(method='ffill')
arima_model = ARIMA(flu_cases, order=(5,1,0))
model_fit = arima_model.fit()
predictions = model_fit.forecast(steps=30)
plt.plot(flu_cases, label="Actual")
plt.plot(pd.date_range(flu_cases.index[-1], periods=30, freq='D'), predictions, label="Forecast")
plt.legend()
plt.show()

# Classification Model - Predicting High-Risk Patients
df["High_Risk"] = (df["Severity_Level"] == "High").astype(int)
features = ["Waiting_Time (min)", "Flu_Cases", "Hour"]
X = df[features]
y = df["High_Risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

