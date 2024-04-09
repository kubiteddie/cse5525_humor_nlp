'''
Tests the model created by TF-IDF_for_QA.py
Can be used for another model if loaded correctly
'''

import joblib

# Load the saved model and vectorizer
model = joblib.load('datastore/humor_classifier.pkl')
vectorizer = joblib.load('datastore/vectorizer.pkl')

# Function for testing
def predict_humor(question, answer):
    combined_text = question + " " + answer
    transformed_text = vectorizer.transform([combined_text])
    prediction = model.predict(transformed_text)
    return "Humor" if prediction[0] == 1 else "Fact"

# sample question/answer pairs
samples = [
    {"question": "who was the first president of the united states", "answer": "George Washington."},
    {"question": "What do you call a magic dog?", "answer": "A labracadabrador."}
]

for sample in samples:
    print("Q:", sample["question"])
    print("A:", sample["answer"])
    print("Prediction:", predict_humor(sample["question"], sample["answer"]), "\n")

# User input test
user_question = input("Enter your question: ")
user_answer = input("Enter the answer: ")

prediction_result = predict_humor(user_question, user_answer)
print(f"The question/answer pair is: {prediction_result}")