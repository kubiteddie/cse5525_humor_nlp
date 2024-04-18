import time
print("______________INITIALIZING MODEL______________")
time.sleep(1)
from llama_cpp import Llama
import json

# chatGPT_output = [1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 
#  0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 
#  0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 
#  1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 
#  1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1]

human_results = [1,0,1,0,0,0,1,1,1,0,0,1,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0]

def execute_localLLM(qas):
    normalized_responses = []
    i = 1
    for pair in qas:
        # generate prompt
        message = [{
            "role": "user",
            "content": f"Determine if this question and answer pair is fact or humor. The question is: {pair['question']} The answer is: {pair['answer']}"
        }]

        print("\n______________MODEL GENERATING RESPONSE______________")
        # Generate a response
        response = llm.create_chat_completion(messages=message)["choices"][0]["message"]["content"]
        print("_________________RESPONSE GENERATED__________________\n")

        # Print the response
        print("Response Number ", i)
        print(response)
        
        if "humor" in response:
            normalized_responses.append(True)
        else:
            normalized_responses.append(False)
        i += 1
    return normalized_responses

def LLM_accuracy(LLM_responses, true_values):
    num_correct = 0
    for i in range(len(LLM_responses)):
        if LLM_responses[i] == true_values[i]['humor']:
            num_correct += 1
    return num_correct, num_correct / len(LLM_responses)


# model_path = "C:/Users/edgar/Software/LLMs/text-generation-webui-main/models/wizardlm-1.0-uncensored-llama2-13b.Q5_K_M.gguf"

# llm = Llama(model_path=model_path)
# print("___________INITIALIZATION COMPLETE____________\n")

with open('../datastore/Questions_and_Answers.json', 'r') as f:
    qas = json.load(f)
qas = qas[:30]

# normalized_responses = execute_localLLM(qas)
# print("\n_________________Local LLM execution complete!__________________")

# print("\nResponses from Local LLM normalized to true and false answers:\n", normalized_responses)

num_correct, accuracy = LLM_accuracy(human_results, qas)

print(f"\nLocal LLM wizardlm-1.0-uncensored-llama2-13b uncensored identified {num_correct} out of {len(qas)} question-answer pairs correctly. Accuracy = {accuracy*100:.2f}%")