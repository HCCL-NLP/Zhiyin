import os
import pdb

import torch
import platform
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_dir='/home/nlp/data/Zhiyin-7B-Chat/new'
def init_model():
    print("init model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(
        model_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print("欢迎使用知音大模型，clear 清空历史，CTRL+C 中断生成，stream 流式生成（默认开启），exit 退出。")
    return []


def main(stream=True):
    model, tokenizer = init_model()
    messages = clear_screen()
    while True:
        prompt = input("\nuser：")
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        print("\nassistant：", end='')
        if prompt.strip() == "stream":
            stream = not stream
            print("(流式生成：{})\n".format("开" if stream else "关"), end='')
            continue
        messages.append({"role": "user", "content": prompt})
        if stream:
            position = 0
            try:
                for response in model.chat(tokenizer, messages, stream=True):
                    print(response[position:], end='', flush=True)
                    position = len(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
            except KeyboardInterrupt:
                pass
            print()
        else:
            response = model.chat(tokenizer, messages)
            print(response)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
