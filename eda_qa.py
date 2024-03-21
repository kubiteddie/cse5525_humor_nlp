import json


def printQA(file):
    data = json.load(open(file, encoding='utf8'))
    qas = list()

    for item in data['data']:
        for pg in item['paragraphs']:
            for qa in pg['qas']:
                curr = dict()
                curr['question'] = qa['question']
                curr['answer'] = qa['answers'][0]['text']
                curr['humor'] = False
                qas.append(curr)
                    # f.write("Question: " + qa['question'] + "\n")
                    # f.write("Answer: " + qa['answers'][0]['text'] + "\n\n")
                    # qas.append((qa['question'], qa['answers'][0]['text']))
                
    with open('Questions_and_Answers.json', 'w', encoding='utf8') as f:
        json.dump(qas, f, indent=4)

    return qas

questions = json.load(open('train-v1.1.json', encoding='utf8'))

with open('clean_train.json', 'w') as newfile:
    json.dump(questions, newfile, indent=2)

file = "clean_train.json"
qas = printQA(file)
#print(qas)