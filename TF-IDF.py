'''
Test for TF-IDF
Logistic Regression and Naive bayes model used.
Showing decent accuracy but not really working well on the test sentence. 
Probably overfitting

'''

import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report, f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from load_data import load_data

# Load the CSV data 
file_path = "datastore/dataset.csv"  
data = pd.read_csv(file_path)

data_df = load_data(file_path)

# Split the data into features and target
X = data_df[['question', 'answer']]
y = data_df['humor']

print(len(X))
print(len(y))
# Convert the text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

overall_accuracy = accuracy_score(y_test, y_pred)
print("Overall Accuracy:", overall_accuracy)

f1_score_humor = f1_score(y_test, y_pred, pos_label=1)
print("F1-score for Humor:", f1_score_humor)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
humor_accuracy = tp / (tp + fn)
print("Humor-specific Accuracy:", humor_accuracy)
fact_accuracy = tn / (tn + fp)
print("Fact-specific Accuracy:", fact_accuracy)

# f1 score for joke
f1_joke = f1_score(y_test, y_pred, pos_label=1)
print("F1 Score (Joke):", f1_joke)

# accuracy for the joke
joke_accuracy = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])
print("Accuracy (Joke):", joke_accuracy)

# accuracy for the statement
fact_accuracy = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])
print("Accuracy (Statement):", fact_accuracy)

# Function to classify new sentences with all models
def classify_sentence(sentence):
    sentence_vector = vectorizer.transform([sentence])
    predictions = {
        "Logistic Regression": "Joke" if lr_model.predict(sentence_vector)[0] else "Not a Joke"
    }
    return predictions

new_sentences = [
    "Why couldn't the bike stand up? Because it was two-tired.",
    "The turtle jumped over the homeless guy.",
    "Never hit a man with glasses. Hit him with a baseball bat.",
    "aa",
    "bsdb",
    "My eyes are brown.",
    "I saw a balloon."
]

for sentence in new_sentences:
    classification_results = classify_sentence(sentence)
    print(f"Sentence: {sentence}")
    for model, result in classification_results.items():
        print(f"{model}: {result}")
    print("\n")
