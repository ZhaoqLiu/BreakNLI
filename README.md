# BreakNLI
This project is the experiment of breaking NLI system
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
