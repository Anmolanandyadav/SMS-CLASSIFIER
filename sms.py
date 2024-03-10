# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load data (replace "path/to/your/data.csv" with your actual file path)
data = pd.read_csv("path/to/your/data.csv")

# Separate features (text) and target (label)
sms = data["SMS"]
label = data["Label"]

# Preprocess text (optional, can include more steps like stemming/lemmatization)
def preprocess_text(text):
  text = text.lower()  # Convert to lowercase
  text = text.replace("[^a-zA-Z0-9 ]", "")  # Remove special characters
  return text

sms = sms.apply(preprocess_text)

# Vectorize text using count vectorizer
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(sms)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict on new SMS (replace "new_sms" with your message)
new_sms = "This is a new message to classify"
new_features = vectorizer.transform([new_sms])
prediction = model.predict(new_features)

if prediction[0] == "spam":
  print("The message is classified as spam")
else:
  print("The message is classified as non-spam")
