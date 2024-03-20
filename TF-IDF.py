'''
Test for TF-IDF
Logistic Regression and Naive bayes model used.
Showing decent accuracy but not really working well on the test sentence. 
Probably overfitting

'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

# Load the CSV data 
file_path = "C:\\Users\\eodyd\\Downloads\\archive\\dataset.csv"  # Path to the file
data = pd.read_csv(file_path)

# Split the data into features and target
X = data['text']
y = data['humor']

# Convert the text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Initialize and train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
nb_pred = nb_model.predict(X_test)

print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred):.2f}")
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, nb_pred):.2f}")

# Function to classify new sentences with all models
def classify_sentence(sentence):
    sentence_vector = vectorizer.transform([sentence])
    predictions = {
        "Logistic Regression": "Joke" if lr_model.predict(sentence_vector)[0] else "Not a Joke",
        "Naive Bayes": "Joke" if nb_model.predict(sentence_vector)[0] else "Not a Joke"
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
