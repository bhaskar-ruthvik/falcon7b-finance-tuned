# Falcon-7B Model Fine-Tuned on Finance Data

This is a Falcon-7b model fine-tuned on finance data by using Parameter Efficient Fine-Tuning(PEFT) and Quantized Low-Rank Adapters. The data includes stock prices, transaction data, tweets about stocks and their sentiment analysis, frequently asked questions about the Finance industry.

## How was the data created?
* The stock data was taken from the YFinance library which provides up-to-date stock prices. As the model cannot handle realtime data the last known stock prices will be dated 11-07-2023 which is the date of training of the model
* The user transaction data was hand-crafted by a group member to look at the possibilities it would unlock without having to deal with the privacy concerns of using real user data
* The tweets and their sentiments were taken from a kaggle dataset by Rutvik Nelluri
* The faqs about finance were noted down through research on the internet and the prompts were framed using that data

## Why was the training data so small?
* Despite the fact that the actual data collected was of a very large scale including atleast 5000 datapoints for the stocks and the tweets analysis, and that similar prompts for the transaction data and faqs could be generated using the OpenAI, the decision to stick with a much smaller subset of the data is to improve the training time on lower-end GPUs
* This is also why PEFT and QLoRA have been used for the fine-tuning of the model which drastically reduce the trainable weights from 7 Billion to 432k which is significantly smaller

## How was the model trained?
The model was trained by using the built-in transformers trainer with max_steps set to 140 which is approximately equal to 4 epochs of training. The final step training loss was 0.49.

## How to Run? 
1. First run the cells to install all the libraries in their required versions (This code snippet uses a custom version of transformers but we can now use the official release):

``` python
  pip install -Uqqq pip --progress-bar off
  pip install -qqq bitsandbytes==0.39.0 --progress-bar off  
  pip install -qqq torch==2.0.1 --progress-bar off  
  pip install -qqq -U git+https://github.com/huggingface/transformers.git@e03a9cc --progress-bar off
  pip install -qqq -U git+https://github.com/huggingface/peft.git@42a184f
  pip install -qqq -U git+https://github.com/huggingface/accelerate.git@c9fbb71 --progress-bar off
  pip install -qqq datasets==2.12.0 --progress-bar off
  pip install -qqq loralib==0.1.1 --progress-bar off
  pip install -qqq einops==0.6.1 --progress-bar off
```
2. Now import all the necessary libraries and set the default device to the gpu:
``` python
import json
import os
from pprint import pprint

import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```
3. Then load the model using 8-bit preferably if the hardware allows for it, to speed up the inference time:

``` python
PEFT_MODEL = 'bhaskar-ruthvik/falcon7b-finance-tuned'

config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict = True,
    device_map = 'auto',
    trust_remote_code = True,
    load_in_8bit = True
)

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model = PeftModel.from_pretrained(model,PEFT_MODEL)
```

4. Setup the generation configuration:
``` python
generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
```

5. Now declare a function to generate prompts so the code can be reused
``` python
def generate_response(question: str)->str:
  prompt = f"""
  <human>: {question}
  <bot>:
  """.strip()
  encoding = tokenizer(prompt,return_tensors='pt').to(device)
  with torch.inference_mode():
    outputs = model.generate(input_ids=encoding.input_ids,
                           attention_mask = encoding.attention_mask,
                           generation_config = generation_config)

  response = tokenizer.decode(outputs[0],skip_special_tokens=True)

  assistant_start = "<bot>:"
  response_start = response.find(assistant_start)
  return response[response_start+ len(assistant_start) :].strip()
```

6. Now provide the prompt to the model and wait for the inference (takes about 40 seconds):
``` python
prompt = 'What is estate planning?'
print('%.300s' % generate_response(prompt))
```

### This Repository is also available on [HuggingFace.](https://huggingface.co/bhaskar-ruthvik/falcon7b-finance-tuned)
