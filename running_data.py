import pandas as pd
from load_data import load_data

def blackbox(x, y):
    print(type(x))
    print(type(y))

    print(y)

filepath = "Questions_and_Answers.json"

data_df = load_data(filepath)

X = data_df[['question', 'answer']]
y = data_df['humor']

blackbox(X, y)