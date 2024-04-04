import json
import pandas as pd

def load_data(json_file):
    column_names = ['question', 'answer', 'humor']
    f = open('datastore/Questions_and_Answers.json')
    data = json.load(f)
    rows = []

    for pair in data:
        row = [pair["question"], pair["answer"], pair["humor"]]
        rows.append(row)

    df = pd.DataFrame(data=rows, columns=column_names)
    print(df)
    f.close()
    return df