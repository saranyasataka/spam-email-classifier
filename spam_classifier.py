import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1','v2']]  # keep only label and message
df.columns = ['label','message']

# Convert labels to 0 (ham) and 1 (spam)
df['label_num'] = df.label.map({'ham':0, 'spam':1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# Convert text to numbers
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Predict
y_pred = model.predict(X_test_counts)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test with your own message
test_msg = ["Congratulations!"]
test_count = vectorizer.transform(test_msg)
print("Prediction (1=Spam,0=Ham):", model.predict(test_count))
