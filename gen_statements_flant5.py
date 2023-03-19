import time
import torch
import argparse
import numpy as np

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import *

# generate contradicting statement of hypothesis using Flan-T5 of huggingface
def gen_statements_flant5(prems, hypos, prompt, num_generation, flant5, tokenizer, verbose=True):
    input_texts = [prompt.replace('<premise>', prem).replace('<hypothesis>', hypo)\
                    for prem, hypo in zip(prems, hypos)]

    if verbose:
        print('Generating using prompt: %s' %prompt)
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)
    
    generation_config = GenerationConfig(
        num_beams=num_generation, 
        temperature=0.6,
        num_return_sequences=num_generation
    )

    # generation_config = GenerationConfig(
    #     do_sample=True,
    #     top_p=0.92,
    #     top_k=20,
    #     temperature=0.4,
    #     num_return_sequences=num_generation
    # )

    start = time.time()
    outputs = flant5.generate(**inputs, generation_config=generation_config)
    end = time.time()
    
    generated_texts = np.array(tokenizer.batch_decode(outputs, skip_special_tokens=True)).reshape(-1, num_generation)
    if verbose:
        print('Generation for %d hypotheses completed!' %(len(generated_texts))) 
        print('%.4fs for each generation' %((end-start)/len(input_texts)))

    del flant5, inputs, outputs
    torch.cuda.empty_cache()
    return generated_texts

def generate_raw_statements(dataset, lan_model, 
                            gen_prompt, batch_size,
                            data_dir,
                            generation_size=10, 
                            num_inst_each_class=None,
                            model_saved_path=None):
    if len(gen_prompt) == 2:
        generate_statement_by_label = True
    else:
        generate_statement_by_label = False

    print('Generation prompt:', gen_prompt)
    entail_idx = np.where(np.array(dataset['label']) == 0)[0]
    contrad_idx = np.where(np.array(dataset['label']) == 2)[0]
    
    cut_idx = min(len(entail_idx), len(contrad_dix)) if num_inst_each_class is None else num_inst_each_class
    print('Generating hypothesis using model %s with %d instances for each class (0 & 2)' %(lan_model, cut_idx))
    select_idx = np.append(entail_idx[:cut_idx], contrad_idx[:cut_idx])

    dataset['premise'] = dataset['premise'][select_idx]
    dataset['hypothesis'] = dataset['hypothesis'][select_idx]
    dataset['label'] = dataset['label'][select_idx]

    premises = batch_it(dataset['premise'], batch_size=batch_size, keep_tail=True)
    hypotheses = batch_it(dataset['hypothesis'], batch_size=batch_size, keep_tail=True)
    labels = batch_it(dataset['label'], batch_size=batch_size, keep_tail=True)
    
    lan_model = 'google/' + lan_model
    if model_saved_path is not None:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_saved_path).to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(lan_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(lan_model)
    
    generated_text_set = None
    for batch, (prem, hypo, label) in tqdm(enumerate(zip(premises, hypotheses, labels)),
                           total=len(labels), desc='Processing batches'):
        generated_texts = gen_statements_flant5(prem, hypo, gen_prompt,
                              generation_size, model,
                              tokenizer, label, verbose=False)
        if generated_text_set is None:
            generated_text_set = generated_texts
        else:
            generated_text_set = np.append(generated_text_set, generated_texts, axis=0)
    
    dataset['g_hypotheses'] = generated_text_set
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, 'mnli_g.pickle'), 'wb') as f:
        pickle.dump(dataset, f)
    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test of contradiction generation using FlanT5')
    parser.add_argument('-n', '--name', type=str, default='mnli', 
                        choices=['mnli', 'snli', 'anli', 'qnli'], 
                        help='name of the NLI dataset')
    parser.add_argument('-s', '--size', type=int, default=10,
                        help='sample size')
    parser.add_argument('-gm', '--genmodel', type=str, default='flan_t5_base', 
                        choices=['flan_t5_base', 'flan_t5_large', 'flan_t5_xl'],
                        help='model used to generate contradictive statements')
    parse.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if arg.demo:

        gen_model = args.genmodel.replace('_', '-')
        dataset = fetch_and_organize_data(args.name)

        prompt_texts = [
            'Generate 10 statements contradicting to the following hypothesis. <hypothesis>',
            'Generate a statement contradicting to the following hypothesis. <hypothesis>',
            '<premise> Hypothesis: <hypothesis> Generate a contradictive statement of the hypothesis.'
        ]
        
        np.random.seed(seed=25)
        sampled_index = np.random.choice(np.arange(len(dataset['train']['hypothesis'])), size=args.size)
        sampled_hypos = np.array(dataset['train']['hypothesis'])[sampled_index]
        sampled_prems = np.array(dataset['train']['premise'])[sampled_index]
        sampled_labels = np.array(dataset['train']['label'])[sampled_index]

        prompt_text = 'Generate a statement contradicting to the following statement. <hypothesis>'

        model = 'google/' + model
        flant5 = AutoModelForSeq2SeqLM.from_pretrained(model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model)
        generated_texts = gen_statements_flant5(sampled_prems, sampled_hypos, prompt_text, 10, flant5, tokenizer)
        print('Show example:')
        print('Sample premise: %s' %sampled_prems[0])
        print('Sample hypothesis: %s' %sampled_hypos[0])
        print('Generated contradictive statements:\n', generated_texts[0])

    else:

        lan_model = 'flan-t5-xl'
        gen_prompt = [
            'Generate a statement that contradicts the following statement: <hypothesis>',
            'Rephrase the following sentence while preserving its original meaning: <hypothesis>'
        ]
        batch_size = 16
        data_dir = './data'
        # data_dir = '/content/drive/MyDrive/Thesis/Implementation/data'
        model_saved_path = '/content/drive/MyDrive/Thesis/Implementation/Model'
        dataset = generate_raw_statements(dataset['validation'], lan_model, 
                                          gen_prompt, batch_size,
                                          data_dir,
                                          generation_size=10, 
                                          num_inst_each_class=None,
                                          model_saved_path=model_saved_path)