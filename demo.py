import os
import numpy as np
import random
import time
import datasets
import openai
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def fetch_data(dataset_name, save_to_cur_dic):
    
    dataset_map = {
        'snli': 'snli',
        'anli': 'anli',
        'mnli': 'SetFit/mnli',
        'qnli': 'SetFit/qnli'
    }

    cur_path = os.path.abspath('./')
    data_path = os.path.join(cur_path, 'data')
    if save_to_cur_dic and not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = datasets.load_dataset(dataset_map[dataset_name], cache_dir=data_path)
    return dataset


def standarize_text(texts):
    for i, text in enumerate(texts):
        if not text.strip().endswith(('.', '?')):
            texts[i] = text.strip() + '.'
    return texts
    

def fetch_and_organize_data(data_name='mnli', save_to_cur_dic=False):
    r"""
    Fetch dataset online and save to determined cache.
    
    Args:
        data_name (defaults to 'mnli'): 
            Specify the name of target dataset.
            Splits name of datasets (key1):
                'mnli': 'train', 'test', 'validation';
                'snli': 'train', 'test', 'validation';
                'qnli': 'train', 'test', 'validation';
                'anli': 'train', 'test', 'dev' with suffixes of '_r1', '_r2', and '_r3'.
            Features of datasets (key2):
                'mnli': 'text1', 'text2', 'label', 'idx', 'label_text';
                'snli': 'premise', 'hypothesis', 'label';
                'qnli': 'text1', 'text2', 'label', 'idx', 'label_text';
                'anli': 'uid', 'premise', 'hypothesis', 'label', 'reason'.
            Labels of datasets:
                'mnli': -1-Not labeled, 0-Entailment, 1-Neutral, 2-Contradiction;
                'snli': 0-Entailment, 1-Neutral, 2-Contradiction;
                'qnli': -1-Not labeled, 1-Entailment, 1-Contradiction;
                'anli': 0-Entailment, 1-Neutral, 2-Contradiction.
                
        save_to_cur_dic (defaults to False): 
            Whether to save the downloaded data to the current 
            dictionary. If False, save to the default cache of
            current environment.
            
    Returns:
        A dict of splits 'train', 'test', and 'validation' with
        a dict of 'premise', 'hypothesis', or 'label' for each split.
    """
    
    print('Fetching dataset of {}'.format(data_name))
    dataset = fetch_data(data_name, save_to_cur_dic)
    print('Data fetched!')
    
    key1s = ['train', 'test', 'validation']
    key2_p = 'premise' if data_name in ['snli', 'anli'] else 'text1'
    key2_h = 'hypothesis' if data_name in ['snli', 'anli'] else 'text2'
    key2_l = 'label'
    
    returned_dict = dict()
    if data_name == 'anli':
        for key1 in key1s:
            premises, hypotheses, labels = [], [], []
            anli_key1 = 'dev' if key1 == 'validation' else key1
            for key in dataset.keys():
                if key.startswith(anli_key1):
                    data = dataset[key]
                    premises += data[key2_p]
                    hypotheses += data[key2_h]
                    labels += data[key2_l]
            returned_dict[key1] = {
                'premise': standarize_text(premises),
                'hypothesis': standarize_text(hypotheses),
                'label': labels
            }
            
    else:
        for key1 in key1s:
            data = dataset[key1]
            premises = data[key2_p]
            hypotheses = data[key2_h]
            labels = data[key2_l]
            returned_dict[key1] = {
                'premise': standarize_text(premises),
                'hypothesis': standarize_text(hypotheses),
                'label': labels
            }
            
    print('Obtained full data of {}, including:\n\
           {} instances in training set\n\
           {} instances in test set\n\
           {} instances in validation set\n'.format(
                data_name, len(returned_dict['train']['premise']),
                len(returned_dict['test']['premise']),
                len(returned_dict['validation']['premise']))
         )
        
    return returned_dict


def gen_contradiction_gpt3(hypos, prompt, model="text-davinci-003"):
    if '<premise>' in prompt and '<hypothesis>' in prompt:
        input_texts = [prompt.replace('<premise>', prem).replace('<hypothesis>', hypo)\
                       for prem, hypo in zip(prems, hypos)]
    else:
        input_texts = [' '.join([prompt, hypo]) for hypo in hypos]
    
    start = time.time()
    openai.api_key = '<API-key>'
    response = openai.Completion.create(model=model, prompt=input_texts, temperature=0, max_tokens=512)
    # 150,000 tokens/min, 200 tokens per instance
    end = time.time()
    
    generated_texts = []
    for inst in response['choices']:
        generated_text = []
        try:
            for sent in inst['text'].strip().split('\n'):
                generated_text.append(sent.split('. ')[1])
        except Exception as e:
            continue
        generated_texts.append(generated_text)
        
    print('%.4fs for each generation with model: %s' %((end-start)/len(input_texts), model))
    return np.array(generated_texts)


def gen_contradiction_flant5(prems, hypos, prompt, model='flan-t5-base'):
    model = 'google/' + model
    flant5 = AutoModelForSeq2SeqLM.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    if '<premise>' in prompt and '<hypothesis>' in prompt:
        input_texts = [prompt.replace('<premise>', prem).replace('<hypothesis>', hypo)\
                       for prem, hypo in zip(prems, hypos)]
    else:
        input_texts = [' '.join([prompt, hypo]) for hypo in hypos]
        
    # print('Input texts: ', input_texts)
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
    
    start = time.time()
    outputs = flant5.generate(**inputs, max_length=256)
    end = time.time()
    
    generated_texts = np.array(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    print('contradictions: ', generated_texts)
    print('%.4fs for each generation with model: %s' %((end-start)/len(input_texts), model))
    return generated_texts


def nli_predict(model, tokenizer, statements):
    inputs = tokenizer(statements, return_tensors='pt', padding=True)
    outputs = model.generate(**inputs)
    pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    pred = np.where(np.array(pred)=='No', 2, 0)
    return pred


def demo(dataset, model, tokenizer, gen_cont_prompt, nli_prompt, func_gen_cont, k):
    
    # randomly select k instances from the given dataset
    # and generate 10 contradictions for each hypothesis.
    sampled_index = np.random.choice(np.arange(len(dataset['train']['hypothesis'])), size=k)
    sampled_hypos = np.array(dataset['train']['hypothesis'])[sampled_index]
    print(sampled_hypos)
    sampled_prems = np.array(dataset['train']['premise'])[sampled_index]
    sampled_labels = np.array(dataset['train']['label'])[sampled_index]
    generated_texts = func_gen_cont(sampled_prems, sampled_hypos, gen_cont_prompt)
    
    total_suc_cont = 0
    total_suc_break = 0
    for prem, hypo, gen_hypos, label in zip(sampled_prems, sampled_hypos, generated_texts, sampled_labels):
        # select contradictions recognized by NLI system
        input_text = [nli_prompt.replace('<premise>', hypo).replace('<hypothesis>', gen_hypo) for gen_hypo in gen_hypos]
        prediction = nli_predict(flant5, tokenizer, input_text)
        suc_cont = np.where(prediction==2)[0]
        total_suc_cont += len(suc_cont)
        
        # examine how many generated statements change the original NLI label
        cont_gen_hypos = gen_hypos[suc_cont]
        if len(cont_gen_hypos) == 0:
            total_suc_break += 0
        else:
            break_text = [nli_prompt.replace('<premise>', prem).replace('<hypothesis>', gen_hypo) for gen_hypo in cont_gen_hypos]
            break_pred = nli_predict(flant5, tokenizer, break_text)
            suc_break = np.where(break_pred != label)[0]
            total_suc_break += len(suc_break)
    
    perc_suc_cont = total_suc_cont / (10 * len(sampled_prems))
    perc_suc_break = total_suc_break / total_suc_cont
    print('%.4f%% of generated statements are labeled contradicting to corresponding hypothesis.' %(perc_suc_cont*100))
    print('%.4f%% of the generated contradictions could change the original label.' %(perc_suc_break*100))


if __name__ == '__main__':

	dataset = fetch_and_organize_data('anli', save_to_cur_dic=True)
	model = 'google/flan-t5-base'
	flant5 = AutoModelForSeq2SeqLM.from_pretrained(model)
	tokenizer = AutoTokenizer.from_pretrained(model)
	prompt_text = 'Generate 10 statements contradicting to the following hypothesis.'

	prompt1 = '<premise> Based on the paragraph above, can we conclude that <hypothesis>?'
	prompt2 = '<premise> Can we infer the following? <hypothesis>'
	prompt3 = 'Read the following and determine if the hypothesis can be inferred from the premise: Premise: <premise> Hypothesis:<hypothesis>'

	demo(dataset, flant5, tokenizer, prompt_text, prompt1, gen_contradiction_gpt3, 20)