import argparse

import torch
from sklearn.metrics import f1_score

from utils import *
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def evaluate_nli_model(model_name, data_name, prompt, batch_size=128):

    dataset = fetch_and_organize_data(data_name, True)

    val_premises = dataset['validation']['premise']
    val_hypotheses = dataset['validation']['hypothesis']
    val_labels = dataset['validation']['label']

    print('Evaluating model %s with %s dataset ...' %(model_name, data_name))
    print('NLI prompt: %s' %prompt)
    input_text = [prompt.replace('<premise>', prem).replace('<hypothesis>', hypo) \
                  for prem, hypo in zip(val_premises, val_hypotheses)]
    batched_input = batch_it(input_text, batch_size)

    model_path = 'google/' + model_name
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    pred_ids = []
    for batch in batched_input:
        pred_ids += list(nli_predict(model, tokenizer, batch))

    score = f1_score(val_labels[:len(pred_ids)], pred_ids, average='macro')
    print('Evaluation completed!')
    print('Macro F1 score: %.4f' %score)

    del model
    torch.cuda.empty_cache()
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of NLI systems')
    parser.add_argument('-n', '--name', type=str, default='mnli', 
                        choices=['mnli', 'snli', 'anli', 'qnli'], 
                        help='name of the NLI dataset')
    parser.add_argument('-m', '--model', type=str, default='flan_t5_base', 
                        choices=['flan_t5_base', 'flan_t5_large', 'flan_t5_xl'], 
                        help='evaluated model')
    parser.add_argument('-p', '--prompt', type=int, default=2, 
                        choices=[0, 1, 2], 
                        help='evaluated prompt')
    args = parser.parse_args()

    dataset = args.name
    model = args.model.replace('_', '-')
    prompt = args.prompt

    nli_prompts = [
        '<premise> Based on the paragraph above, can we conclude that <hypothesis>?',
        '<premise> Can we infer the following? <hypothesis>',
        'Read the following and determine if the hypothesis can be inferred from the premise: Premise: <premise> Hypothesis: <hypothesis>'
    ]

    evaluate_nli_model(model, 'mnli', nli_prompts[prompt])

