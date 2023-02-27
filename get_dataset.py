import os
import argparse

import datasets


def fetch_data(dataset_name, save_to_cur_dic=True):
    
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

    print('Loading dataset of', dataset_name)
    dataset = datasets.load_dataset(dataset_map[dataset_name], cache_dir=data_path)
    print('Dataset loaded and saved to', data_path)
    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download NLI dataset')
    parser.add_argument('-n', '--name', type=str, default='mnli', 
                        choices=['mnli', 'snli', 'anli', 'qnli'], 
                        help='name of the NLI dataset')
    args = parser.parse_args()
    fetch_data(args.name)