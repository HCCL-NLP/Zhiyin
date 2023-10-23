# Zhiyin
## Zhiyin-Chat


知音聊天机器人由中科院声学所语音与智能信息处理实验室研发，用于回答用户问题和提供信息，以帮助人们解决问题和获取知识。本项目后续会开放指令训练数据、相关模型、训练代码、应用场景等。

## 📝 项目主要内容

### 🚀 推理代码

详见[cli_demo.py]，尽可能简化的一个推理代码实现。

### 🤖 模型

详见[Zhiyin/models](models/)
, 以及我们实验室研发的大模型[Zhiyin-7B-Chat](https://huggingface.co/HCCL-NLP/Zhiyin-7B-Chat)。

## 安装依赖
```shell
pip install -r requirements.txt
```

## Python使用范例

```python
>>> import torch
>>> from transformers.generation.utils import GenerationConfig
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> model_dir='HCCL-NLP/Zhiyin-7B-Chat'#离线使用时改为模型储存路径
>>> tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)
>>> model.generation_config = GenerationConfig.from_pretrained(model_dir)
>>> messages = []
>>> messages.append({"role": "user", "content": "诸葛亮北伐失败的原因"})
>>> response = model.chat(tokenizer, messages)
>>> print(response)
诸葛亮北伐失败的原因有很多，以下是一些主要因素：

1. 资源不足：尽管蜀国在诸葛亮的领导下进行了积极的备战，但与强大的魏国相比，其资源和实力仍然相差甚远。

2. 地理劣势：魏国占据着中原地区，地势平坦、易守难攻；而蜀国则地处西南，地势险要但运输困难。

3. 战略失误：诸葛亮北伐时，曾试图攻打魏国的首都洛阳，但这一战略被许多历史学家认为是错误的，因为这样会陷入敌人的腹地，难以持。

4. 人才流失：蜀国在诸葛亮北伐期间，人才流失严重，尤其是蜀汉的精英将领和士兵，这大大削弱了蜀国的实力。

5. 时间有限：诸葛亮北伐的时间有限，他在五丈原与司马懿相持数月后，因病去世，北伐计划被迫中止。

6. 内部矛盾：蜀国内部存在一定的矛盾和冲突，如诸葛亮与李严的关系紧张等，这些矛盾对北伐产生了一定的影响。
```
