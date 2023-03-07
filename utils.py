import numpy as np
from tqdm import tqdm

import torch

from get_dataset import *


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def standarize_text(texts):
    for i, text in enumerate(texts):
        if not text.strip().endswith(('.', '?')):
            texts[i] = text.strip() + '.'
    return np.array(texts)
            
def fetch_and_organize_data(data_name='mnli', save_to_cur_dic=True):
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
                'label': np.array(labels)
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
                'label': np.array(labels)
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

def text_label_to_id(text_labels):
    map_dict = {
      'yes': 0,
      'no': 2
    }
    return np.array([map_dict.get(text.lower(), 1) for text in text_labels])


def batch_it(data, batch_size, keep_tail=False):
    batched_data = []
    upper = int(len(data) / batch_size) + 1 if keep_tail else int(len(data) / batch_size)
    for i in range(upper):
        batched_data.append(data[i * batch_size: (i+1) * batch_size])
        
    if len(batched_data[-1]) == 0:
        batched_data = batched_data[:-1]
        
    if not keep_tail:
        batched_data = np.array(batched_data)
        
    return batched_data


def nli_predict(model, tokenizer, premises, 
                hypotheses, prompt, batch_size, 
                description=None):
    
    if prompt is not None:
        texts = [prompt.replace('<premise>', prem).replace('<hypothesis>', hypo)
                      for prem, hypo in zip(premises, hypotheses)]
    batched_texts = batch_it(texts, batch_size=batch_size, keep_tail=True)
    
    predictions = None
    for batch in tqdm(batched_texts, desc=description):
        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors='pt', padding='longest').to(device)
            outputs = model.generate(**inputs)
        pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        pred_ids = text_label_to_id(pred)
        
        if predictions is None:
            predictions = pred_ids
        else:
            predictions = np.append(predictions, pred_ids)
        
        del inputs, outputs
        torch.cuda.empty_cache()
    
    return predictions


def sample_result(results, idx=None):
    if idx is None:
        idx = np.random.randint(0, len(results[0]))
    selected_conts, break_preds, samples = results
    sampled_prems, sampled_hypos, sampled_labels = samples

    print('Original primise:', sampled_prems[idx])
    print('Original hypothesis:', sampled_hypos[idx])
    print('Original label:', sampled_labels[idx])
    print('Selected contradictions:', selected_conts[idx])
    print('Generated labels:', break_preds[idx])


def evaluate_results(label, preds, generate_ent_for_con):
    ine, s_ine = 0, 0
    
    if generate_ent_for_con:
        label = 0
        
    for p in preds:
        if label in p:
            ine += 1
        if label in p or 1 in p:
            s_ine += 1
            
    return ine, s_ine


def show_results(labels, preds, con_con):
    ent_idx = np.where(labels==0)[0]
    ine_ent, s_ine_ent = evaluate_results(0, preds[ent_idx], con_con)
    print('Class of Entailment:')
    print('%.2f%% is inequal triangle; %.2f%% is strictly inequal triangle.' %(
        ine_ent/len(ent_idx)*100, s_ine_ent/len(ent_idx)*100))
    
    con_idx = np.where(labels==2)[0]
    ine_con, s_ine_con = evaluate_results(2, preds[con_idx], con_con)
    print('Class of Entailment:')
    print('%.2f%% is inequal triangle; %.2f%% is strictly inequal triangle.' %(
        ine_con/len(con_idx)*100, s_ine_con/len(con_idx)*100))
    
    num_label = len(labels)
    print('Overall results:')
    print('%.2f%% is inequal triangle; %.2f%% is strictly inequal triangle.' %(
        (ine_ent+ine_con)/num_label*100, (s_ine_ent+s_ine_con)/num_label*100))