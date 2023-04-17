import time

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import *


# generate contradicting statement of hypothesis using Flan-T5 of huggingface
def gen_statements_flant5(prems, hypos, prompt, num_generation, flant5, tokenizer, verbose=True):
    input_texts = [prompt.replace('<premise>', prem).replace('<hypothesis>', hypo) \
                   for prem, hypo in zip(prems, hypos)]

    if verbose:
        print('Generating using prompt: %s' % prompt)
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True).to(device)

    generation_config = GenerationConfig(
        num_beams=num_generation,
        temperature=0.6,
        num_return_sequences=num_generation
    )

    # generation_config = GenerationConfig(
    #    do_sample=True,
    #    top_p=0.92,
    #    top_k=20,
    #    temperature=0.4,
    #    num_return_sequences=num_generation
    # )

    start = time.time()
    outputs = flant5.generate(**inputs, generation_config=generation_config)
    end = time.time()

    generated_texts = np.array(tokenizer.batch_decode(outputs, skip_special_tokens=True)).reshape(-1, num_generation)
    if verbose:
        print('Generation for %d hypotheses completed!' % (len(generated_texts)))
        print('%.4fs for each generation' % ((end - start) / len(input_texts)))

    del flant5, inputs, outputs
    torch.cuda.empty_cache()
    return generated_texts


def generate_raw_statements(dataset_name, lan_model,
                            gen_prompts, batch_size,
                            data_dir,
                            generation_size=10,
                            model_saved_path=None,
                            placeholder='<PH>'):
    print('Generation prompt:', gen_prompts)
    print('Dataset:', dataset_name)

    dataset = fetch_and_organize_data(dataset_name, True)['validation']

    for key, value in dataset.items():
        dataset[key] = dataset[key]

    premises = batch_it(dataset['premise'], batch_size=batch_size, keep_tail=True)
    hypotheses = batch_it(dataset['hypothesis'], batch_size=batch_size, keep_tail=True)

    lan_model = 'google/' + lan_model
    model_load_path = lan_model if model_saved_path is None else model_saved_path
    model = AutoModelForSeq2SeqLM.from_pretrained(model_load_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(lan_model)

    repeat_idx = []
    for key, gen_prompt in gen_prompts.items():
        generated_text_set = None
        jdx = 0
        for batch, (prem, hypo) in tqdm(enumerate(zip(premises, hypotheses)),
                                        total=len(premises), desc='Processing batches'):
            generated_texts = np.hstack((
                gen_statements_flant5(prem, hypo, gen_prompt[0],
                                      generation_size, model,
                                      tokenizer, verbose=False),
                gen_statements_flant5(prem, hypo, gen_prompt[1],
                                      generation_size, model,
                                      tokenizer, verbose=False)
            ))

            for i, generation_set in enumerate(generated_texts):
                unique_sents = list(set(generation_set))
                if hypo[i] in unique_sents:
                    unique_sents.remove(hypo[i])

                if len(unique_sents) < generation_size:
                    repeat_idx.append(jdx)
                    unique_sents = np.array([placeholder for _ in range(generation_size)])
                else:
                    random_idx = np.random.choice(np.arange(len(unique_sents)), size=generation_size)
                    unique_sents = np.array(unique_sents)[random_idx]
                jdx += 1

                if generated_text_set is None:
                    generated_text_set = unique_sents[np.newaxis, ...]
                else:
                    generated_text_set = np.append(generated_text_set, unique_sents[np.newaxis, ...], axis=0)
            torch.cuda.empty_cache()

        dataset['g_hypotheses_{}'.format(key[:3])] = generated_text_set

    full_idx = np.arange(len(dataset['label']))
    for idx in repeat_idx:
        full_idx = np.delete(full_idx, np.where(full_idx == idx))
    print(
        '%d instances were deprecated because of repetitive generation.' % (len(dataset['label']) - full_idx.shape[0]))
    for key, value in dataset.items():
        dataset[key] = dataset[key][full_idx]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, '{}_g.pickle'.format(dataset_name)), 'wb') as f:
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
        print('Sample premise: %s' % sampled_prems[0])
        print('Sample hypothesis: %s' % sampled_hypos[0])
        print('Generated contradictive statements:\n', generated_texts[0])

    else:

        lan_model = args.genmodel.replace('_', '-')
        # lan_model = 'flan-t5-xl'
        gen_prompt = {
            'negative': ['Generate a statement that contradicts the following statement: <hypothesis>',
                         '<hypothesis> Generate a statement that contradict the meaning of the last sentence.'],
            'positive': ['Rephrase the following sentence while preserving its original meaning: <hypothesis>',
                         '<hypothesis> Rewrite the last sentence while preserving its original meaning.']
        }
        batch_size = 8
        data_dir = './data'
        # data_dir = '/content/drive/MyDrive/Thesis/Implementation/data'
        model_saved_path = '/content/drive/MyDrive/Thesis/Implementation/Model'
        dataset = generate_raw_statements('mnli', lan_model,
                                          gen_prompt, batch_size,
                                          data_dir,
                                          generation_size=10,
                                          model_saved_path=model_saved_path)
