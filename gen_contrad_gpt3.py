import time
import argparse
import numpy as np

import openai
from utils import fetch_and_organize_data


apikey = '<API-KEY>'

# generate contradicting statement of hypothesis using openai GPT-3 API
def gen_contradiction_gpt3(prems, hypos, prompt, api_key, model="text-davinci-003"):
    input_texts = [prompt.replace('<premise>', prem).replace('<hypothesis>', hypo)\
                   for prem, hypo in zip(prems, hypos)]
    
    print('Generating using prompt: %s' %prompt)
    start = time.time()
    openai.api_key = api_key
    response = openai.Completion.create(model=model, prompt=input_texts, temperature=0, max_tokens=512)
    # 150,000 tokens/min, 200 tokens per instance
    end = time.time()
    
    generated_texts = []
    for inst in response['choices']:
        generated_text = []
        try:
            exception_idx = max(inst['text'].find('\n\n', 3), 0)
            for sent in inst['text'][exception_idx:].strip().split('\n'):
                generated_text.append(sent.split('. ')[1])
        except Exception as e:
            continue
        generated_texts.append(generated_text)

    print('Generation for %d hypotheses completed!' %(len(generated_texts)))    
    print('%.4fs for each generation with model: %s' %((end-start)/len(input_texts), model))
    return np.array(generated_texts)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test of contradiction generation using GPT3')
    parser.add_argument('-n', '--name', type=str, default='mnli', 
                        choices=['mnli', 'snli', 'anli', 'qnli'], 
                        help='name of the NLI dataset')
    parser.add_argument('-s', '--size', type=int, default=10,
                        help='sample size')
    args = parser.parse_args()

    dataset = fetch_and_organize_data(args.name)

    # np.random.seed(seed=23)
    sampled_index = np.random.choice(np.arange(len(dataset['train']['hypothesis'])), size=args.size)
    sampled_hypos = np.array(dataset['train']['hypothesis'])[sampled_index]
    sampled_prems = np.array(dataset['train']['premise'])[sampled_index]
    sampled_labels = np.array(dataset['train']['label'])[sampled_index]

    prompt_text = 'Generate 10 statements contradicting to the following statement. <hypothesis>'

    generated_texts = gen_contradiction_gpt3(sampled_prems, sampled_hypos, prompt_text, apikey)
    print('Show example:')
    print('Sample premise: %s' %sampled_prems[0])
    print('Sample hypothesis: %s' %sampled_hypos[0])
    print('Generated contradictive statements:\n', generated_texts[0])