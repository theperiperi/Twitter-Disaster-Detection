import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the train and test datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Data preprocessing
# Remove NaN values
train_data = train_data.dropna(subset=['tweet', 'disaster'])
test_data = test_data.dropna(subset=['tweet'])

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train = vectorizer.fit_transform(train_data['tweet'])
X_test = vectorizer.transform(test_data['tweet'])
y_train = train_data['disaster']

# Train a classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
# (Optional) Split train data for validation
# X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# model.fit(X_train_split, y_train_split)
# y_pred = model.predict(X_val)
# print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# Predict on test data
test_predictions = model.predict(X_test)
test_data['predicted_disaster'] = test_predictions

# Post-processing and analysis
# (Optional) Convert predicted_disaster to readable labels (e.g., 'Disaster' or 'Not Disaster')

# Output predictions
test_data[['tweet', 'predicted_disaster']].to_csv("predictions.csv", index=False)
