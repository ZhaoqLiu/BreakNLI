import numpy as np
import scipy
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
                'qnli': -1-Not labeled, 0-Entailment, 1-not entailment;
                'anli': 0-Entailment, 1-Neutral, 2-Contradiction.
                
        save_to_cur_dic (defaults to False): 
            Whether to save the downloaded data to the current 
            dictionary. If False, save to the default cache of
            current environment.
            
    Returns:
        A dict of splits 'train', 'test', and 'validation' with
        a dict of 'premise', 'hypothesis', or 'label' for each split.
    """
    suffix = None
    if '_' in data_name:
        suffix = data_name[-3:]
        data_name = data_name[:-3]
    
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
            if suffix is not None:
                anli_key1 += suffix
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


def batch_it(data, batch_size, keep_tail=False):
    batched_data = []
    upper = int(len(data) / batch_size) + 1 if keep_tail else int(len(data) / batch_size)
    for i in range(upper):
        batched_data.append(data[i * batch_size: (i + 1) * batch_size])

    if len(batched_data[-1]) == 0:
        batched_data = batched_data[:-1]

    if not keep_tail:
        batched_data = np.array(batched_data)

    return batched_data


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


def evaluate_results(label, preds, positive_statement):
    ine, r_ine = 0, 0

    if label == 1:
        for p in preds:
            if not np.all(p == label):
                r_ine += 1
        return r_ine, r_ine
    
    if positive_statement:
        label = 2 * int(not bool(label))
        
    for p in preds:
        if label in p:
            ine += 1
        if label in p or 1 in p:
            r_ine += 1
            
    return ine, r_ine


def show_results(labels, preds, positive_statement):

    perc_rec = []
    total_ine, total_r_ine = 0, 0
    label_to_text = {0: 'Entailment', 1: 'Neutrality', 2: 'Contradiction'}
    for l in [0, 1, 2]:
        corres_idx = np.where(labels==l)[0]
        if len(corres_idx) == 0:
            continue
        ine, r_ine = evaluate_results(l, preds[corres_idx], positive_statement)
        perc_ine, perc_r_ine = np.round(ine/len(corres_idx)*100, 2), np.round(r_ine/len(corres_idx)*100, 2)
        print('Class of %s' % label_to_text[l])
        print('%.2f%% inequal triangle; %.2f%% relaxed inequal triangle.' %(
          perc_ine, perc_r_ine))
        total_ine += ine
        total_r_ine += r_ine
        perc_rec.append(str(perc_ine)+'%/'+ str(perc_r_ine)+'%')
      
    num_label = len(labels)
    perc_ine, perc_r_ine = np.round(total_ine/num_label*100, 2), np.round(total_r_ine/num_label*100, 2)
    print('Overall:')
    print('%.2f%% inequal triangle; %.2f%% relaxed inequal triangle.' %(
        perc_ine, perc_r_ine))
    perc_rec.append(str(perc_ine)+'%/'+ str(perc_r_ine)+'%')

    return perc_rec


def ks_divergence(v1, v2):
    return np.max(np.abs(v1 - v2))

def calculate_distance(mat1, mat2, metric):
    if metric == 'cosine_distance':
        func = scipy.spatial.distance.cosine
    elif metric == 'ks_divergence':
        func = ks_divergence
    elif metric == 'js_divergence':
        dist = np.array([scipy.spatial.distance.jensenshannon(
            v1, v2, base=2) for v1, v2 in zip(mat1, mat2)])
        return np.array([d if not np.isnan(d) else 0 for d in dist])
    else:
      return np.array([scipy.stats.entropy(v1, v2, base=2) for v1, v2 in zip(mat1, mat2)])
    return np.array([func(v1, v2) for v1, v2 in zip(mat1, mat2)])

def measure(dataset, metric, sorted):
    if metric == 'cosine_distance':
      mat1 = dataset['h_embeddings']
      mat2 = dataset['ha_embeddings']
    else:
      mat1 = scipy.special.softmax(dataset['h_logits'], axis=-1)
      mat2 = scipy.special.softmax(dataset['ha_logits'], axis=-1)
      if sorted:
        print(sorted)
        mat1.sort()
        mat2.sort()

      if metric == 'variance':
          return (np.round(np.mean(mat1.std(axis=-1)), 4), mat2.std(axis=-1))

    num_inst, num_generation, dim = mat2.shape

    mat1 = np.repeat(mat1, num_generation, axis=0)
    mat2 = mat2.reshape(mat1.shape)

    dist = calculate_distance(mat1, mat2, metric).reshape((-1, num_generation))
    return dist

def overall_analysis(dataset, metric, sorted=False):
    # metric: 'cosine_distance', 'kl_divergence', 'ks_divergence', 'variance'
    dist = measure(dataset, metric, sorted=sorted)

    unchange, change = [], []
    if metric == 'variance':
        (ori_vari, dist) = dist

    for i, label in enumerate(dataset['labels']):
        for d, p_label in zip(dist[i], dataset['pred_labels'][i]):
            if label == p_label:
                unchange.append(d)
            else:
                change.append(d)
    dist_unchange = np.round(np.mean(unchange), 4)
    dist_change = np.round(np.mean(change), 4)

    if metric == 'variance':
        return ori_vari, dist_unchange, dist_change
    else:
        return dist_unchange, dist_change