# BreakNLI
This project concerns the evaluation of NLI systems. The setting could be formulated as follows: 
1. Given an NLI dataset $D= \lbrace(x_1, y_1),...,(x_n, y_n) \rbrace$, where we have the premise-hypothesis pair $x_i=\lbrace p_i, h_i\rbrace$ and the label $y_i \in \lbrace Entailment, Contradiction, Neutral \rbrace$, and an NLI model $M$, where $M(p_i, h_i)=y_i$.
2. Using `Flan-T5-xl` (referred to $G$), generate 5 statements that contradicts the hypothesis, namely $G(h_i)=\lbrace h_i^1, ...,h_i^k,..., h_i^5 \rbrace$ and $M(h_i,h_i^k)=Contradiction$.
3. Evaluate whether the following 3 triangles hold or not by $M(p_i, h_i^k)$.
![Image text](imgs/triangles.png)

**Our hypothesis:**  
&ensp;&ensp; If the system is not able to change the label of an example accordingly, then the predictions are based on shallow patterns as opposed to a deep language understanding.

**NLI models:**
* google/flan-t5-base (tested)
* google/flan-t5-large (tested)
* google/flan-t5-xl (tested)
* google/flan-t5-xxl
* facebook/bart-large-mnli
* roberta-large-mnli
* valhalla/distilbart-mnli-12-1
* microsoft/deberta-base-mnli
* microsoft/deberta-large-mnli
* microsoft/deberta-xlarge-mnli

**Problems to be looked into:**  
1. For pair $(p_i,h_i)$ whose $M(p_i, h_i)=Contradiction$ (Triangle 2), the generation of contradictive statements is hard for the current way. Because two sentences could contradict each other in many aspects, inducing legit inequality of the triangles. The following is a typical example where $M(p, h) = M(h, h^k)=M(p,h^k)=Contradiction$.
    > premise: The house is surprisingly small and simple, with one bedroom, a tiny kitchen, and a couple of social rooms.  
    > hypothesis: The house is very large and boasts over ten bedrooms, a huge kitchen, and a full sized olympic pool.  
    > generated hypothesis: The house is very small and boasts over ten bedrooms, a huge kitchen, and a full sized olympic pool.
    > 
2. Generation parameters of $G$ needs further consideration.
3. For the cases that $M(p, h^k)$ fails to comply the relationship indicated by the triangles (1 & 3), does it really break the NLI system? We need to inspect the generated statements.

**Temporary results:**
| Inequal/strictly inequal | Flan-T5-base  | Flan-T5-large | Flan-T5-xl   |
| ------------------------ | :-----------: | :-----------: | :----------: |
| Entailment               | 15.61%/20.34% | 13.13%/20.05% | 8.83%/16.82% |
| Contradiction            | 6.49%/12.19%  | 4.41%/9.95%   | 3.01%/7.09%  |
| Overall                  | 11.03%/16.24% | 8.72%/14.93%  | 5.83%/11.81% |

Inequal: At least 1 (among 5) $M(p, h^k)$ is given the opposite label of the label that it should have been predicted. For instance, given $M(p,h)=Contradiction$ and $M(h, h^k)=Entailment$, $M(p, h^k)=Entailment$.  
Strictly inequal: At least 1 (among 5) $M(p, h^k)$ does not comply with the relationship indicated by the corresponding triangle. For instance, given $M(p,h)=Contradiction$ and $M(h, h^k)=Entailment$, $M(p, h^k)=Entailment\mid Neutral$.  
Entailment: Results for premise-hypothesis pairs whose $M(p, h)=Entailment$.  

## Requirements
The environment should meet the following requirements:
  ```markdown
  datasets==2.2.2
  numpy==1.23.4
  openai==0.26.5
  scikit_learn==1.2.1
  torch==1.11.0+cu113
  transformers==4.10.3
  ```
  or install the required package by
  ```sh
  pip install -r requirements.txt
  ```
  
## Installation
Clone the repo 
  ```sh
  git clone https://github.com/ZhaoqLiu/BreakNLI.git
  ```

## Usage
### evaluate_nli.py
Run `evaluate_nli.py` to evaluate the NLI system. There are several optional parameters that control the experiment: `-n` specifies the used dataset (defaults to `mnli`), `-m` specifies the tested NLI model (defaults to `flan_t5_base`), and `-p` specifies the indice of the prompt to be used (defaults to `2`). For example:
  ```sh
  python evaluate_nli.py -n mnli -m flan_t5_base -p 2
  ```
The following is the result of different NLI system (variants of Flan-T5) using different prompts. The winning prompt is 
  ```markdown
  Read the following and determine if the hypothesis can be inferred from the premise: Premise: <premise> Hypothesis: <hypothesis>
  ```
|          | base   | large  | xl     |
|:--------:|:------:|:------:|:------:|
| prompt_1 | 0.7954 | 0.8527 | 0.8911 |
| prompt_2 | 0.8261 | 0.8751 | 0.8987 |
| prompt_3 | 0.8338 | 0.8884 | 0.9070 |

### get_dataset.py
Run `get_dataset.py` to download the NLI dataset and save to the current directory. Use parameter `-n` to specify the dataset to be downloaded (defaults to `mnli`).
  ```sh
  python get_dataset.py -n anli
  ```
### gen_contrad_\*.py
Run `gen_contrad_flant5.py` and `gen_contrad_gpt3.py` to test the contradiction generation using Flan-T5 and GPT3 API, respectively. The premise-hypothesis pair will be randomly selected from the dataset specified by parameter `-n` (defaults to `mnli`) with the size specified by `-s` (defaults to `10`). Use parameter `-gm` to select the tested model of Flan-T5 (defaults to `flan_t5_base`). Note that you have to specify your API key of openai in the file before you use GPT3, which could be generated from https://platform.openai.com/account/api-keys.
  ```sh
  python gen_contrad_flant5.py -n mnli -s 10 -gm flan_t5_base
  python gen_contrad_gpt3.py -n mnli -s 10
  ```
  
### evaluation.py
Run `evaluation.py` to run a demo for the experiment of breaking NLI system. Similarly, use `-n` and `-s` to define the NLI dataset and sample size, `-gm` to select generation model, and `-nm` to select the model performing NLI. By default, the demo will sample 10 instances from mnli dataset using GPT3 for generation and Flan-T5-base for NLI.
  ```sh
  python evaluation.py -n mnli -s 10 -gm gpt3 -nm flan_t5_base
  ```

## Examples

### Example 1 - Contradiction could happen at different aspects of the sentence

Original primise: For example, the most recent edition of the Unified Agenda (April 2000) describes <ins>4,441 rulemaking actions</ins> under development or recently completed by 60 federal departments and agencies.

Original hypothesis: <ins>All the development</ins> happened with the individual federal departments and agencies.

Original label: 2

Selected contradictions: 

'All the development happened with the individual state departments and agencies.'

'There are no federal departments and agencies.'
 
'There was no development at all.'
 
'Federal departments and agencies are not involved in the development.'
 
'There are no federal departments and agencies that are involved in development.'
 
Generated labels: [2 2 2 2 2]

### Example 2 

Original primise: The alligator farm here is something of a curiosity.

Original hypothesis: The turtle farm is something of an oddity.

Original label: 2

Selected contradictions:

'The turtle farm is a common sight.'

'It is a normal farm.'

'No one has ever seen a turtle farm before'

'There are many turtle farms.' "It's a normal farm."

 'Nobody knows anything about the turtle farm.'
 
 "It's a normal place."
 
 'The turtle farm is a normal place.'
 
Generated labels: [2 2 2 2 2 2 2 2]

### Example 3 - Nonsense generated statements

Original primise: You were going to say?

Original hypothesis: You finished your previous sentence.

Original label: 2

Selected contradictions:

'There are no people in the room.'

"We can't do anything about it."

 'Nobody knows how to swim.'
 
Generated labels: [2 2 2]

### Example 4 - Nonsense daily conversation

Original primise: same and uh we'll.

Original hypothesis: I disagree.

Original label: 2

Selected contradictions:

'I am a believer.'

'I agree with you.'

'I am a believer in God.'

 "It's impossible to tell the difference between the two."
 
 "It's impossible for me to believe that." 'I agree with him.'
 
Generated labels: [1 0 1 1 2 1]

### Example 5 - Legit inequality

Original primise: These tentacled creatures may look terrible under water, but once out in the air they're revealed as small and not dangerous.

Original hypothesis: They look less threatening when they are out of the water.

Original label: 0

Selected contradictions:

'They look more threatening when they are in the water.'
 
 ...
 
'When they are in the water they look more dangerous.']
 
Generated labels: [0 ... 0]
