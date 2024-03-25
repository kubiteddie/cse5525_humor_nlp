import json
import pandas as pd

column_names = ['question', 'answer', 'humor']

f = open('Questions_and_Answers.json')

data = json.load(f)

rows = []
for i in data:
    row = [i.question, i.answer, i.humor]
    rows.append(row)

df = pd.DataFrame(data=rows, columns=column_names)

print(df)

f.close()