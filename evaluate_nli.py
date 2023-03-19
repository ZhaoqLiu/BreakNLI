import argparse

import torch
from sklearn.metrics import accuracy_score, f1_score

from utils import *
from models import NLIModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def evaluate_nli_model(nli_model, data_name, nli_prompt=None, batch_size=128, metric='accuracy'):

    dataset = fetch_and_organize_data(data_name, True)

    val_premises = dataset['validation']['premise']
    val_hypotheses = dataset['validation']['hypothesis']
    val_labels = dataset['validation']['label']

    model = NLIModel(nli_model, nli_prompt)

    print('Evaluating model %s with %s dataset ...' %(nli_model, data_name))
    if nli_prompt is not None:
        print('NLI prompt: %s' %nli_prompt)
    nli_preds = model.predict(val_premises, val_hypotheses, batch_size, description='Evaluating NLI on (p,h)')

    if metric == 'f1':
        score = f1_score(val_labels, nli_preds, average='macro')
    elif metric == 'accuracy':
        score = accuracy_score(val_labels, nli_preds)
    print('Evaluation completed!')
    print('%s score: %.4f' %(metric, score))

    del model
    torch.cuda.empty_cache()
    return score, val_labels, nli_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of NLI systems')
    parser.add_argument('-n', '--name', type=str, default='mnli', 
                        choices=['mnli', 'snli', 'anli', 'qnli'], 
                        help='name of the NLI dataset')
    parser.add_argument('-nm', '--nlimodel', type=str, default='flan_t5_base', 
                        choices=['flan_t5_base', 'flan_t5_large', 'flan_t5_xl',
                        'flan_t5_xxl', 'bart_large_mnli', 'roberta_large_mnli',
                        'distilbart_mnli_12-1', 'deberta_base_mnli', 
                        'deberta_large_mnli', 'deberta_xlarge_mnli'],
                        help='model used to perform NLI')
    args = parser.parse_args()

    dataset = args.name
    model = args.nlimodel.replace('_', '-')

    nli_prompt = 'Read the following and determine if the hypothesis can be inferred from the premise: Premise: <premise> Hypothesis: <hypothesis>'

    evaluate_nli_model(model, dataset, nli_prompt)

