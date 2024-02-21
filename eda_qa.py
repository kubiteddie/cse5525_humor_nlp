import json

questions = json.load(open('train-v1.1.json', encoding='utf8'))

with open('clean_train.json', 'w') as newfile:
    json.dump(questions, newfile, indent=2)