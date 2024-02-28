import json


def printQA(file):
    data = json.load(open(file, encoding='utf8'))
    qas = list()

    with open('Questions_and_Answers.txt', 'w', encoding='utf8') as f:
        for item in data['data']:
            for pg in item['paragraphs']:
                for qa in pg['qas']:
                    f.write("Question: " + qa['question'] + "\n")
                    f.write("Answer: " + qa['answers'][0]['text'] + "\n\n")
                    qas.append((qa['question'], qa['answers'][0]['text']))
    return qas

questions = json.load(open('train-v1.1.json', encoding='utf8'))

with open('clean_train.json', 'w') as newfile:
    json.dump(questions, newfile, indent=2)

file = "clean_train.json"
qas = printQA(file)
print(qas)