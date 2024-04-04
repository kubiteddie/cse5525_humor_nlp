import json
import random

def printQA(qafile, jokefile):
    data = json.load(open(qafile, encoding='utf8'))
    qas = list()

    for item in data['data']:
        for pg in item['paragraphs']:
            for qa in pg['qas']:
                curr = dict()
                curr['question'] = qa['question']
                curr['answer'] = qa['answers'][0]['text']
                curr['humor'] = False
                qas.append(curr)

    data2 = json.load(open(jokefile, encoding='utf8'))
    for joke in data2:        
        if len(joke['body']) > 255:
            currjoke = dict()
            currjoke['question'] = joke['title']
            currjoke['answer'] = joke['body']
            currjoke['humor'] = True
            qas.append(currjoke)

    return qas

questions = json.load(open('train-v1.1.json', encoding='utf8'))

with open('clean_train.json', 'w') as newfile:
    json.dump(questions, newfile, indent=2)

qafile = "datastore/clean_train.json"
jokefile = 'datastore/reddit_jokes.json'
qas = printQA(qafile, jokefile)
random.seed(83)
random.shuffle(qas)
id=0

for item in qas:
    item['id'] = id
    id += 1

with open('datastore/Questions_and_Answers.json', 'w', encoding='utf8') as f:
    json.dump(qas, f, indent=4)
