import argparse
import numpy as np

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from gen_contrad_gpt3 import gen_contradiction_gpt3
from gen_contrad_flant5 import gen_contradiction_flant5
from utils import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"
api_key = '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'


def demo(dataset, nli_model, gen_model, gen_prompt, nli_prompt, sample_size, generation_size):
    
    # randomly select sample_size instances from the given dataset
    # and generate 10 contradictions for each hypothesis.
    ignore_neutral_idx = np.where(np.array(dataset['train']['label']) == 0)[0]
    
    sampled_index = np.random.choice(np.arange(len(ignore_neutral_idx)), size=sample_size)
    sampled_hypos = dataset['train']['hypothesis'][ignore_neutral_idx][sampled_index]
    sampled_prems = dataset['train']['premise'][ignore_neutral_idx][sampled_index]
    sampled_labels = dataset['train']['label'][ignore_neutral_idx][sampled_index]

    if gen_model == 'gpt3':
        generated_texts = gen_contradiction_gpt3(sampled_prems, sampled_hypos, gen_prompt, api_key)
    else:
        generated_texts = gen_contradiction_flant5(sampled_prems, sampled_hypos, gen_prompt, generation_size, model=gen_model)
    torch.cuda.empty_cache()

    print('Performing NLI using prompt:', nli_prompt)
    nli_model = 'google/' + nli_model
    NLI_model = AutoModelForSeq2SeqLM.from_pretrained(nli_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(nli_model)
    
    total_suc_cont = 0
    total_label_change = 0
    total_suc_break = 0
    selected_cont = []
    break_preds = []
    for prem, hypo, gen_hypos, label in zip(sampled_prems, sampled_hypos, generated_texts, sampled_labels):
        # select contradictions recognized by NLI system
        input_text = [nli_prompt.replace('<premise>', hypo).replace('<hypothesis>', gen_hypo) for gen_hypo in gen_hypos]
        prediction = nli_predict(NLI_model, tokenizer, input_text)
        suc_cont = np.where(prediction==2)[0]
        total_suc_cont += len(suc_cont)
        
        # examine how many generated statements change the original NLI label
        cont_gen_hypos = gen_hypos[suc_cont]
        selected_cont.append(cont_gen_hypos)
        if len(cont_gen_hypos) == 0:
            total_label_change += 0
            break_preds.append([])
        else:
            break_text = [nli_prompt.replace('<premise>', prem).replace('<hypothesis>', gen_hypo) for gen_hypo in cont_gen_hypos]
            break_pred = nli_predict(NLI_model, tokenizer, break_text)
            break_preds.append(break_pred)
            label_change = np.where(break_pred != label)[0]
            total_label_change += len(label_change)
            
            label_unchange = np.where(break_pred == label)[0]
            if len(label_unchange) != 0:
                total_suc_break += 1
            
    
    perc_suc_cont = total_suc_cont / (10 * len(sampled_prems))
    perc_label_change = total_label_change / total_suc_cont
    perc_suc_break = total_suc_break / len([item for item in selected_cont if len(item)!=0])
    print('%.2f%% of generated statements are labeled contradicting to corresponding hypothesis.' %(perc_suc_cont*100))
    print('%.2f%% of the generated contradictions could change the original label.' %(perc_label_change*100))
    print('We successfully broke %.2f%% of the instances.' %(perc_suc_break*100))
    del NLI_model
    torch.cuda.empty_cache()
    
    return selected_cont, break_preds, [sampled_prems, sampled_hypos, sampled_labels]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Test of contradiction generation using GPT3')
    parser.add_argument('-n', '--name', type=str, default='mnli', 
                        choices=['mnli', 'snli', 'anli', 'qnli'], 
                        help='name of the NLI dataset')
    parser.add_argument('-s', '--size', type=int, default=10,
                        help='sample size')
    parser.add_argument('-gm', '--genmodel', type=str, default='gpt3', 
                        choices=['gpt3', 'flan_t5_base', 'flan_t5_large', 'flan_t5_xl'],
                        help='model used to generate contradictive statements')
    parser.add_argument('-nm', '--nlimodel', type=str, default='flan_t5_base', 
                        choices=['flan_t5_base', 'flan_t5_large', 'flan_t5_xl'],
                        help='model used to perform NLI tests')
    args = parser.parse_args()

    dataset = fetch_and_organize_data(args.name)

    gen_model = args.genmodel.replace('_', '-')
    nli_model = args.nlimodel.replace('_', '-')

    gen_prompt = 'Generate a statement contradicting to the following statement. <hypothesis>'
    # nli_prompt = '<premise> Based on the paragraph above, can we conclude that <hypothesis>?'
    # nli_prompt = '<premise> Can we infer the following? <hypothesis>'
    nli_prompt = 'Read the following and determine if the hypothesis can be inferred from the premise: Premise: <premise> Hypothesis: <hypothesis>'

    results = demo(dataset, nli_model, gen_model, gen_prompt, nli_prompt, args.size, 10)

    sample_results(results)