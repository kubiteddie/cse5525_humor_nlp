import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from datetime import datetime

# Load the data
file_path = 'datastore/Questions_and_Answers.json'

with open(file_path, 'r') as file:
    data = json.load(file)

# Combine question and answer into a single string, create labels
texts = [entry['question'] + " " + entry['answer'] for entry in data]
labels = [1 if entry['humor'] else 0 for entry in data]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

vocabulary_size = 50000

# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=vocabulary_size)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Support Vector Machine": LinearSVC(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10)
}

writeline = "datetime:{}, modeltype:{}, vocabulary size:{}, acc:{}, f1:{}, humoracc:{}, factacc:{}"
outfile = "datastore/modelscores.txt"

# Train, predict, and evaluate each model
for name, model in models.items():
    print(f"model: {name}")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    overall_accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")

    f1_score_humor = f1_score(y_test, y_pred, pos_label=1)
    print(f"F1-score for Humor: {f1_score_humor:.4f}")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    humor_accuracy = tp / (tp + fn)
    print(f"Humor-specific Accuracy: {humor_accuracy:.4f}")
    fact_accuracy = tn / (tn + fp)
    print(f"Fact-specific Accuracy: {fact_accuracy:.4f}")
    print("\n")

    with open(outfile, 'a+') as evalwrite:
        evalwrite.write(writeline.format(datetime.now(), name, vocabulary_size, overall_accuracy, f1_score_humor, humor_accuracy, fact_accuracy))
        evalwrite.write('\n')

with open(outfile, 'a+') as evalwrite:
    evalwrite.write('\n')
    evalwrite.write('\n')
