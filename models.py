import numpy as np
import torch
from tqdm import tqdm

from utils import batch_it

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class NLIModel:
    def __init__(self, model, prompt=None):
        '''
        google/flan-t5-base
        google/flan-t5-large
        google/flan-t5-xl
        google/flan-t5-xxl
        facebook/bart-large-mnli
        roberta-large-mnli
        valhalla/distilbart-mnli-12-1
        microsoft/deberta-base-mnli
        microsoft/deberta-large-mnli
        microsoft/deberta-xlarge-mnli
        '''
        path_prefix = {
            'flan': 'google/',
            'bart': 'facebook/',
            'roberta': '',
            'distilbart': 'valhalla/',
            'deberta': 'microsoft/'
        }

        self.model_name = model
        self.model_path = path_prefix[model.split('-')[0]] + model
        
        if self.model_name.startswith('flan'):
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.prompt = prompt
            
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
    def predict(self, premises, hypotheses, batch_size, description, return_embeddings=False, label_for_qnli=False, return_logits=False):
        predictions, embeddings, logits = [], [], []

        if self.model_name.startswith('flan'):
            texts = [self.prompt.replace('<premise>', prem).replace('<hypothesis>', hypo)
                     for prem, hypo in zip(premises, hypotheses)]
            batched_texts = batch_it(texts, batch_size=batch_size, keep_tail=True)
            
            for batch in tqdm(batched_texts, total=len(batched_texts), desc=description):
                with torch.no_grad():
                    inputs = self.tokenizer(batch, return_tensors='pt', padding='longest').to(device)
                    outputs = self.model.generate(**inputs)
                pred_ids = self._convert_output_to_ids(outputs, label_for_qnli)

                predictions += list(pred_ids)
                
                del inputs, outputs
                torch.cuda.empty_cache()
        
        else:
            batched_prems = batch_it(premises, batch_size=batch_size, keep_tail=True)
            batched_hypos = batch_it(hypotheses, batch_size=batch_size, keep_tail=True)
            
            for prem, hypo in tqdm(zip(batched_prems, batched_hypos), total=len(batched_prems), desc=description):
                prem, hypo = list(prem), list(hypo)
                with torch.no_grad():
                    inputs = self.tokenizer(prem, hypo, return_tensors='pt', padding='longest').to(device)
                    outputs = self.model(**inputs, output_hidden_states=return_embeddings)
                pred_ids = self._convert_output_to_ids(outputs, label_for_qnli)
                
                predictions += list(pred_ids)
                
                if return_logits:
                    logit = outputs.logits.cpu().detach().numpy()
                    for l in logit:
                        logits.append(list(l))

                if return_embeddings:
                    with torch.no_grad():
                        inputs = self.tokenizer(hypo, return_tensors='pt', padding='longest').to(device)
                        embs = self.model.base_model(**inputs).last_hidden_state
                    for emb in embs[:,0,:].cpu().detach().numpy():
                        embeddings.append(list(emb))

                del inputs, outputs
                torch.cuda.empty_cache()
        return np.array(predictions), np.array(embeddings), np.array(logits)
    
    def _convert_output_to_ids(self, outputs, label_for_qnli):
        if self.model_name.startswith('flan'):
            text_to_id_dict = {
                'yes': 0,
                'no': 2,
            }
            pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_ids = np.array([text_to_id_dict.get(text.lower(), 1) for text in pred])
            
        else:
            if label_for_qnli:
                pred_ids = outputs.logits[:,[1,2]].softmax(dim=-1)[:, 1].cpu()
                pred_ids = np.array([int(not(i>0.5)) for i in pred_ids])
            else:
                cls_label = ['entailment', 'neutral', 'contradiction']
                label2id_dict = {key.lower(): value for key, value in self.model.config.label2id.items()}
                label_converter = [label2id_dict[label] for label in cls_label]
                pred_ids = outputs.logits.argmax(-1).cpu().numpy()
                pred_ids = np.array([label_converter[l] for l in pred_ids])

        return pred_ids
    
    def _label_for_qnli(self, pred_ids):
        return np.array([int(bool(l)) for l in pred_ids])
