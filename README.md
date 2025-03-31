# Benchmarking gemma2-9b Efficiency Across Fine-Tuning Techniques

<!-- ![Cute roots](asset/assest1.svg) -->
<!--![c](asset/frontbanner.png)-->
<!--![](bannerforfinetuningproj.jpg)-->
![](asset/config-workflow3.jpg)


## Contents
- [Benchmarking gemma2-9b Efficiency Across Fine-Tuning Techniques](#benchmarking-gemma2-9b-efficiency-across-fine-tuning-techniques)
  - [Contents](#contents)
    - [Dataset Config Workflow](#dataset-config-workflow)
    - [Experiments list](#experiments-list)
  - [Inspiration](#inspiration)

### Dataset Config Workflow

About dataset manipulation:
```python
max_sequence_length = 4096 # after processing data tokenize with this length on text

# on dataloader
batch_size = 80
num_worker = 4
max_token = 1024*16 # when collate the text with the logic of Variable length Collator

```

Data prepared preview:
```json
{
'batch_size': 3, 
'input_ids': tensor([[     2,    106,   1645,  ...,    603, 235292,  15695]]), 
'position_ids': tensor([[   0,    1,    2,  ..., 1948, 1949, 1950]]), 
'seq_lens': [298, 687, 1951], 
'cu_seqlens': tensor([   0,  298,  985, 2936], dtype=torch.int32), 
'max_seq_len': 1951, 
'label': tensor([[1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.]]), 

'input_text': ['<bos><start_of_turn>user\nPlease act as an impartial judge and evaluate the quality of the responses provided by two\nAI assistants to the user question displayed below. You should choose the assistant that\nfollows the user’s instructions and answers the user’s question better. Your evaluation\nshould consider factors such as the helpfulness, relevance, accuracy, depth,   ------------------------ progress and adjusting hyperparameters, such as learning rate and regularization, can help mitigate overfitting.\n<|The End of Conversation between a User and two Assistants|><end_of_turn>\n<start_of_turn>model\nverdict is: [['
```

details about Data loader from training dataset file have some this features mentioned below
```python
for batch in tqdm(dataloader):
    for micro_batch in batch:
        # print(micro_batch)
        print(len(micro_batch['input_ids'][0]))
        print(len(micro_batch['position_ids'][0]))
        print(micro_batch['cu_seqlens'])
        print(len(micro_batch['cu_seqlens']))
        print(len(micro_batch['input_text']))
        print(''.join(micro_batch['input_text'][0].split()[-10:])) # last 10 words
        print("------------------------------")
```

Output shows sample
``` json
13316
13316
tensor([    0,  4330,  5251,  6259,  6564,  9577, 10569, 11543, 13001, 13316],
       dtype=torch.int32)
10
9
betweenaUserandtwoAssistants|><end_of_turn><start_of_turn>modelverdictis:[[
------------------------------
16007
16007
tensor([    0,  3097,  3392,  3836,  4784,  5131,  5468,  6478,  6849,  8687,
         8992,  9545, 10015, 10286, 11698, 12612, 13436, 14668, 16007],
       dtype=torch.int32)
19
18
betweenaUserandtwoAssistants|><end_of_turn><start_of_turn>modelverdictis:[[
------------------------------
```


### Model Architecture



```json
GemmaForCausalLM(
  (model): GemmaModel(
    (embed_tokens): Embedding(256000, 3072, padding_idx=0)
    (layers): ModuleList(
      (0-27): 28 x GemmaDecoderLayer(
        (self_attn): GemmaSdpaAttention(
          (q_proj): Linear(in_features=3072, out_features=4096, bias=False)
          (k_proj): Linear(in_features=3072, out_features=4096, bias=False)
          (v_proj): Linear(in_features=3072, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=3072, bias=False)
          (rotary_emb): GemmaRotaryEmbedding()
        )
        (mlp): GemmaMLP(
          (gate_proj): Linear(in_features=3072, out_features=24576, bias=False)
          (up_proj): Linear(in_features=3072, out_features=24576, bias=False)
          (down_proj): Linear(in_features=24576, out_features=3072, bias=False)
          (act_fn): PytorchGELUTanh()
        )
        (input_layernorm): GemmaRMSNorm()
        (post_attention_layernorm): GemmaRMSNorm()
      )
    )
    (norm): GemmaRMSNorm()
  )
  (lm_head): Linear(in_features=3072, out_features=256000, bias=False)
)

```

### Experiments list

- [ ] document full chart and finesetup and inference setup on local
- [ ] Model architecture:
  - [ ] Gemma2-9b LLMs from scratch for small dataset
  - [ ] alternative for Collators used in model its have some limitations
  - [ ] https://www.kaggle.com/code/emiz6413/inference-gemma-2-9b-4-bit-qlora
  - [ ] https://www.kaggle.com/code/emiz6413/training-gemma-2-9b-4-bit-qlora-fine-tuning?scriptVersionId=187770530
  
<!-- 
  - [ ] Gemma 2 Fine Tuning for Dummies (with 16k, 32k,... Context) [Full Tutorial] [link](https://www.youtube.com/watch?v=EE-nEecm3Wo)
  - [ ] Github Gemma fine tune [link](https://github.com/nodematiclabs/gemma-fine-tune)
  - [ ] Copy of MOSLEH_finetune_gemma2_DEMO.ipynb [ colablink](https://colab.research.google.com/drive/1jN0gS1Yu19yQRpyJZ-MKIuVAY4zP13Pt)
  - [ ] fine_tuning_tutorial.ipynb by deepmind on gemma [colab link](https://colab.research.google.com/github/google-deepmind/gemma/blob/main/colabs/fine_tuning_tutorial.ipynb#scrollTo=S5F3fk22Ecod)
  - [ ] explore the list and experiments with gaps: 
  - [ ] Paper read : Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference [Link](https://arxiv.org/html/2403.04132v1)
  - [ ] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena: [Link](https://ar5iv.org/html/2306.05685#A1.F4) 
    - [ ] Evaluating Large Language Models using LLM-as-a-Judge with Amazon Bedrock [link](https://github.com/aws-samples/evaluating-large-language-models-using-llm-as-a-judge/blob/main/Notebooks/evaluating-large-language-models-using-llm-as-a-judge-with-amazon-bedrock.ipynb)
  - [ ] Other paper and model: [lmsys.org/projects](https://lmsys.org/projects/)
  - [ ] Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality [link](https://lmsys.org/blog/2023-03-30-vicuna/) 
- [ ] explore more the pretrained llms on dataset like present on [OpenAI Evals](https://github.com/openai/evals) / [data](https://github.com/openai/evals/tree/main/evals/registry/data)
- [ ] Train model to check the toxic data [toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) 
- [ ] MT-Bench (Multi-turn Benchmark) - [link](https://klu.ai/glossary/mt-bench-eval)
- [ ] AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback [link](https://github.com/tatsu-lab/alpaca_farm)
- [ ] 
- [ ] -->



## Inspiration

- The dataset to explore [Competition](https://www.kaggle.com/competitions/lmsys-chatbot-arena) and multiple sources from Huggin Face
- Some techniques inspired form here [tascj/kaggle-lmsys-chatbot-arena-post](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685) winner's solutions.
