<div align="center">
<h1><strong>GroveMoE</strong></h1>
</div>
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2508.07785-b31b1b.svg)](https://arxiv.org/abs/2508.07785)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/inclusionAI) -->

<p align="center">
🤗 <a href="https://huggingface.co/collections/inclusionAI/grovemoe-68a2b58acbb55827244ef664">Models</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2508.07785">Paper</a> &nbsp&nbsp | &nbsp&nbsp 🔗 <a href="https://github.com/inclusionAI/GroveMoE">Github</a>&nbsp&nbsp

## Overview

GroveMoE is an **open-source** family of large language models developed by the **AGI Center, Ant Group Research**  that introduces **Grove MoE**, a new sparse architecture using **adjugate experts** for dynamic computation allocation.  
With **33 B total parameters** and **3.14–3.28 B active parameters per token**, GroveMoE delivers **state-of-the-art** results across reasoning, mathematics, and code generation while keeping inference costs low.  

<p align="center"><img src="assets/grovemoe.png" width="95%"></p>

---

## Key Highlights
| Feature | Description |
|---------|-------------|
| **Architecture** | Novel **adjugate experts** grouped with ordinary experts; shared computation is executed once, then reused, cutting FLOPs. |
| **Sparse Activation** | 33 B params total, only **3.14–3.28 B active** per token. |
| **Training** | Mid-training + SFT, up-cycled from **Qwen3-30B-A3B-Base**; preserves prior knowledge while adding new capabilities. |
| **Open** | Weights, configs will be fully released under Apache 2.0 upon approval. |

---


## Run GroveMoE

### 🤗 Transformers Quick Start
Transformers is a library of pretrained natural language processing for inference and training. 

The following contains a code snippet illustrating how to use GroveMoE to generate content based on given inputs. 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "inclusionAI/GroveMoE-Inst"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
```

### 🚀 SGLang Quick Start

[SGLang](https://github.com/sgl-project/sglang) is a fast serving framework for large language models and vision language models.
SGLang could be used to launch a server with OpenAI-compatible API service. 

1️⃣ Install Dependencies

First, install transformers:
```shell
cd /src/transformers-4.51.3
pip install .
```
Then, install SGLang:
2. Install SGLang
```shell
cd src/sglang-0.4.6.post5
pip install .
```
2️⃣ Launch the Server

Run the following command to start SGLang:
```shell
python -m sglang.launch_server --model-path inclusionAI/GroveMoE-Inst --port 30000 --context-length 32768
```

3️⃣ Access the API

Once started, the OpenAI-compatible API will be available at `http://localhost:30000/v1`.

Test it with curl:
```shell
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/GroveMoE-Inst",
    "messages": [{"role": "user", "content": "Hello, SGLang!"}]
  }'

```

## Benchmark Results

<p align="center"><img src="assets/results.png" width="95%"></p>

---

## Citation
If you find our work helpful, feel free to give us a cite.
```bibtex
@article{GroveMoE,
title = {GroveMoE: Towards Efficient and Superior MoE LLMs with Adjugate Experts},
author = {Wu, Haoyuan and Chen, Haoxing and Chen, Xiaodong and Zhou, Zhanchao and Chen, Tieyuan and Zhuang, Yihong and Lu, Guoshan and Zhao, Junbo and Liu, Lin and Huang, Zenan and Lan, Zhenzhong and Yu, Bei and Li, Jianguo},
journal = {arXiv preprint arXiv:2508.07785},
year = {2025}
}
```