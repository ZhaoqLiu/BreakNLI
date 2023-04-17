import os
import pickle
import sys

import pandas as pd
import torch

from evaluation import evaluate
from utils import overall_analysis


def main(dataset, test=False):
    drive_path = '/content/drive/MyDrive/Thesis/Implementation'
    if test:
        dataset_path = os.path.join(drive_path, 'data/{}_g.pickle'.format(dataset))
        result_path = os.path.join(drive_path, 'results/test.xlsx'.format(dataset))
        dataset_save_path = os.path.join(drive_path, 'results/test_dict.pickle'.format(dataset))
        log_file = os.path.join(drive_path, 'logs/test.log'.format(dataset))
    else:
        dataset_path = os.path.join(drive_path, 'data/{}_g.pickle'.format(dataset))
        result_path = os.path.join(drive_path, 'results/{}.xlsx'.format(dataset))
        dataset_save_path = os.path.join(drive_path, 'results/{}_result_dict.pickle'.format(dataset))
        log_file = os.path.join(drive_path, 'logs/{}.log'.format(dataset))

    log_file = open(log_file, "a+")
    stdout_backup = sys.stdout
    sys.stdout = log_file

    nli_prompt = 'Read the following and determine if the hypothesis can be inferred from the premise: Premise: <premise> Hypothesis: <hypothesis>'
    nli_models_bs = {
        'flan-t5-base': 1024,
        'flan-t5-large': 512,
        'flan-t5-xl': 64,
        'bart-large-mnli': 128,
        'roberta-large-mnli': 1024,
        'distilbart-mnli-12-1': 1024,
        'deberta-base-mnli': 1024,
        'deberta-large-mnli': 512,
        'deberta-xlarge-mnli': 256
    }

    if os.path.exists(result_path):
        result_table = pd.read_excel(result_path, index_col=0)
        result_table = result_table.to_dict('list')
    else:
        result_table = {}
    output_recorder = {}

    for model, bs in nli_models_bs.items():
        if model in result_table:
            continue
        else:
            result_table[model] = []

        for switch in [True]:
            results = evaluate(dataset_path,
                               model, nli_prompt,
                               evaluate_positive_statement=switch,
                               take_correct_nli=True,
                               model_saved_path=None,
                               batch_size=bs,
                               test=test)

            result_table[model] += results[0]

            new_dataset = results[1]
            output_recorder[model] = new_dataset

            metrics = ['cosine_distance', 'kl_divergence', 'ks_divergence', 'variance']
            if model.startswith('flan'):
                result_table[model] += ['-' for _ in metrics]
            else:
                for metric in metrics:
                    res = overall_analysis(new_dataset, metric)
                    res_str = '/'.join([str(number) for number in res])
                    result_table[model].append(res_str)

            torch.cuda.empty_cache()

        df = pd.DataFrame(data=result_table)
        df.to_excel(result_path)

    with open(dataset_save_path, 'wb') as f:
        pickle.dump(output_recorder, f)

    log_file.close()
    sys.stdout = stdout_backup
    print('Finished!')


if __name__ == '__main__':
    main('mnli', test=True)
