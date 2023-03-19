import torch
import numpy as np
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
        
    def predict(self, premises, hypotheses, batch_size, description):
        if self.model_name.startswith('flan'):
            texts = [self.prompt.replace('<premise>', prem).replace('<hypothesis>', hypo)
                     for prem, hypo in zip(premises, hypotheses)]
            batched_texts = batch_it(texts, batch_size=batch_size, keep_tail=True)
            
            predictions = None
            for batch in tqdm(batched_texts, total=len(batched_texts), desc=description):
                with torch.no_grad():
                    inputs = self.tokenizer(batch, return_tensors='pt', padding='longest').to(device)
                    outputs = self.model.generate(**inputs)
                pred_ids = self._convert_output_to_ids(outputs)
                if predictions is None:
                    predictions = pred_ids
                else:
                    predictions = np.append(predictions, pred_ids)
                
                del inputs, outputs
                torch.cuda.empty_cache()
        
        else:
            batched_prems = batch_it(premises, batch_size=batch_size, keep_tail=True)
            batched_hypos = batch_it(hypotheses, batch_size=batch_size, keep_tail=True)
            
            predictions = None
            for prem, hypo in tqdm(zip(batched_prems, batched_hypos), total=len(batched_prems), desc=description):
                prem, hypo = list(prem), list(hypo)
                with torch.no_grad():
                    inputs = self.tokenizer(prem, hypo, return_tensors='pt', padding='longest').to(device)
                    outputs = self.model(**inputs)
                pred_ids = self._convert_output_to_ids(outputs)
                
                if predictions is None:
                    predictions = pred_ids
                else:
                    predictions = np.append(predictions, pred_ids)
                
                del inputs, outputs
                torch.cuda.empty_cache()
                
        return predictions
    
    def _convert_output_to_ids(self, outputs):
        if self.model_name.startswith('flan'):
            text_to_id_dict = {
                'yes': 0,
                'no': 2,
            }
            pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            pred_ids = np.array([text_to_id_dict.get(text.lower(), 1) for text in pred])
            
        else:
            cls_label = ['entailment', 'neutral', 'contradiction']
            label2id_dict = {key.lower(): value for key, value in self.model.config.label2id.items()}
            label_converter = [label2id_dict[label] for label in cls_label]
            pred_ids = outputs.logits.argmax(-1).cpu().numpy()
            pred_ids = np.array([label_converter[l] for l in pred_ids])
        return pred_ids