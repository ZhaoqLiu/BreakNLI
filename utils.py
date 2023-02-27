import numpy as np

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


def batch_it(data, batch_size):
    batched_data = []
    for i in range(int(len(data) / batch_size) - 1):
        batched_data.append(data[i * batch_size: (i+1) * batch_size])
    return batched_data


def nli_predict(model, tokenizer, statements):

    with torch.no_grad():
        inputs = tokenizer(statements, return_tensors='pt', padding='longest').to(device)
        outputs = model.generate(**inputs)
    pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    pred_ids = text_label_to_id(pred)
    del inputs, outputs
    torch.cuda.empty_cache()
    return pred_ids


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
