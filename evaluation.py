import pickle

from models import NLIModel
from utils import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def evaluate(dataset_path, nli_model, nli_prompt,
             evaluate_positive_statement,
             take_correct_nli=True,
             batch_size=32,
             test=False):
    des = 'entailed' if evaluate_positive_statement else 'contradictive'
    print('Evaluating NLI systems %s with generated statements %s to original hypotheses ...' % (nli_model, des))

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    if test:
        for key, value in dataset.items():
            dataset[key] = value[:1000]

    if not evaluate_positive_statement:
        entail_idx = np.where(np.array(dataset['label']) == 0)[0]
        contrad_idx = np.where(np.array(dataset['label']) == 2)[0]
        select_idx = np.append(entail_idx, contrad_idx)
        for key, value in dataset.items():
            dataset[key] = value[select_idx]
        g_hypotheses = dataset['g_hypotheses_neg']
    else:
        g_hypotheses = dataset['g_hypotheses_pos']

    premises = dataset['premise']
    hypotheses = dataset['hypothesis']
    labels = dataset['label']

    model = NLIModel(nli_model, nli_prompt)

    if take_correct_nli:
        # Evaluating (p, h)
        nli_preds = model.predict(
            premises, hypotheses, batch_size,
            return_embeddings=True,
            return_logits=True,
            description='Evaluating NLI on (p,h)'
        )
        (nli_preds, h_embeds, h_logits) = nli_preds
        correct_pred = nli_preds == labels
        correct_nli_idx = np.where(correct_pred)[0]

        premises = premises[correct_nli_idx]
        hypotheses = hypotheses[correct_nli_idx]
        labels = labels[correct_nli_idx]
        g_hypotheses = g_hypotheses[correct_nli_idx]
        if h_embeds.size > 0 and h_logits.size > 0:
            h_embeds = h_embeds[correct_nli_idx]
            h_logits = h_logits[correct_nli_idx]

        num_ent = len(np.where(labels == 0)[0])
        num_con = len(np.where(labels == 2)[0])
        num_neu = len(np.where(labels == 1)[0])
        print('Taking pairs that could be correctly predicted by NLI system, we obtain:')
        print('%d Entailment, %d Contradiction, and %d Neutrality pairs' % (num_ent, num_con, num_neu))

    # Evaluating (h, h')
    temp_hypos = np.array([hypo for hypo in hypotheses for _ in range(10)])
    temp_g_hypos = g_hypotheses.flatten()
    nli_preds, _, __ = model.predict(temp_hypos, temp_g_hypos, batch_size, description='Evaluating NLI on (h,h\')')
    nli_preds = nli_preds.reshape((-1, 10))

    f_idx, f_idj = [], []
    for idx, pred in enumerate(nli_preds):
        s_idx = np.where(pred == 2 * int(not evaluate_positive_statement))[0]

        if len(s_idx) < 5:
            continue
        else:
            f_idj += list(s_idx[:5])
            f_idx.append(idx)

    f_g_idx = [x for x in f_idx for _ in range(5)]
    new_dataset = {
        'premises': premises[f_idx],
        'hypotheses': hypotheses[f_idx],
        'labels': labels[f_idx],
        'g_hypotheses': g_hypotheses[f_g_idx, f_idj].reshape(-1, 5)
    }
    if h_embeds.size > 0 and h_logits.size > 0:
        new_dataset['h_embeddings'] = h_embeds[f_idx]
        new_dataset['h_logits'] = h_logits[f_idx]
    else:
        new_dataset['h_embeddings'], new_dataset['h_logits'] = np.array([]), np.array([])

    num_ent = len(np.where(new_dataset['labels'] == 0)[0])
    num_con = len(np.where(new_dataset['labels'] == 2)[0])
    num_neu = len(np.where(new_dataset['labels'] == 1)[0])
    print('Taking pairs that have more than 5 generated {} statements, we obtain:'.format(des))
    print('%d Entailment, %d Contradiction, and %d Neutrality pairs' % (num_ent, num_con, num_neu))

    # Evaluating (p, h')
    temp_ps = np.array([prem for prem in new_dataset['premises'] for _ in range(5)])
    temp_ghs = new_dataset['g_hypotheses'].flatten()
    nli_preds = model.predict(
        temp_ps, temp_ghs, batch_size,
        return_embeddings=True,
        return_logits=True,
        description='Evaluating NLI on (p,h\')'
    )
    (nli_preds, ha_embeds, ha_logits) = nli_preds
    nli_preds = nli_preds.reshape((-1, 5))
    if ha_embeds.size > 0 and ha_logits.size > 0:
        ha_embeds = ha_embeds.reshape((nli_preds.shape[0], 5, -1))
        ha_logits = ha_logits.reshape((nli_preds.shape[0], 5, -1))

    new_dataset['pred_labels'] = nli_preds
    new_dataset['ha_embeddings'] = ha_embeds
    new_dataset['ha_logits'] = ha_logits

    results = show_results(new_dataset['labels'], nli_preds, evaluate_positive_statement)

    return results, new_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of NLI models')
    parser.add_argument('-nm', '--nlimodel', type=str, default='flan_t5_base',
                        choices=['flan_t5_base', 'flan_t5_large', 'flan_t5_xl',
                                 'flan_t5_xxl', 'bart_large_mnli', 'roberta_large_mnli',
                                 'distilbart_mnli_12-1', 'deberta_base_mnli',
                                 'deberta_large_mnli', 'deberta_xlarge_mnli'],
                        help='model used to perform NLI')
    parser.add_argument('-p', '--positive', type=bool, default=True,
                       help='true if evaluate entailed generation')
    args = parser.parse_args()

    nli_model = args.nlimodel.replace('_', '-')

    data_dir = './data'
    # data_dir = '/content/drive/MyDrive/Thesis/Implementation/data'
    dataset_path = os.path.join(data_dir, 'mnli_g.pickle')
    nli_prompt = 'Read the following and determine if the hypothesis can be inferred from the premise: Premise: <premise> Hypothesis: <hypothesis>'
    results, new_dataset = evaluate(dataset_path, 'roberta-large-mnli',
                                    nli_prompt, evaluate_positive_statement=True,
                                    take_correct_nli=True,
                                    model_saved_path=None,
                                    batch_size=1024,
                                    test=True)
