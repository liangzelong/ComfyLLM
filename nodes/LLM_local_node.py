import os
import time

import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

root_dir = os.environ.get("ComLLM_Path")
# print(os.path.join(root_dir, "support_models.json"))
# with open(os.path.join(root_dir, "support_models.json"), "r") as f:
#     model_maps = json.load(f)["model_maps"]


model_maps={
        "Meta-Llama-3-8B-Instruct": "models/Meta-Llama-3-8B-Instruct",
        "internlm2-chat-1_8b": "models/internlm2-chat-1_8b"
}
model_list = list(model_maps.keys())
def load_model(model_name):
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(root_dir, model_maps[model_name]),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=nf4_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(root_dir, model_maps[model_name]), trust_remote_code=True
    )
    return tokenizer, model


class init_template:
    def __init__(self, model_name):
        self.model_name = model_name
        getattr(self, f"{self.model_name}".replace("-", "_"))()

    def internlm2_chat_1_8b(self):
        self.user_prompt = "<|User|>:{user}\n"
        self.robot_prompt = "<|Bot|>:{robot}<eoa>\n"
        self.cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"

    def Meta_Llama_3_8B_Instruct(self):
        self.user_prompt = (
            "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        )
        self.robot_prompt = (
            "<|start_header_id|>assistant<|end_header_id|>\n\n{robot}<|eot_id|>"
        )
        self.cur_query_prompt = "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def combine_history(model_template, prompt, history=None, if_clear=False):
    templates = init_template(model_template)
    total_prompt = ""

    if history is None or if_clear:
        history = []

    for message in history:
        cur_content = message["content"]
        if message["role"] == "user":
            cur_prompt = templates.user_prompt.replace("{user}", cur_content)
        elif message["role"] == "robot":
            cur_prompt = templates.robot_prompt.replace("{robot}", cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + templates.cur_query_prompt.replace("{user}", prompt)
    return total_prompt


def get_end_token(model_name):
    if model_name == "internlm2-chat-1_8b":
        return "<|im_end|>"
    elif model_name == "Meta-Llama-3-8B-Instruct":
        return "<|eot_id|>"


class Text2Token:
    eot_map = {
        "internlm2-chat-1_8b": "<|eot_id|>",
        "Meta-Llama-3-8B-Instruct": "<|eot_id|>",
    }
    role_map = {
        "internlm2-chat-1_8b": {
            "user": "user",
            "robot": "assistant",
        },
        "Meta-Llama-3-8B-Instruct": {"user": "user", "robot": "system"},
    }

    def __init__(self, model_name, tokenizer):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.eot = self.eot_map.get(self.model_name)
        if self.eot is None:
            raise NotImplementedError(f"{self.model_name} is not implemented")
        self.user = self.role_map[self.model_name]["user"]
        self.robot = self.role_map[self.model_name]["robot"]

    def dialog2message(self, dialog):
        """_summary_
            dilog example
            user:hello!\nrobot:hello\n
        Args:
            dialog (_type_): _description_

        Returns:
            _type_: _description_
        """
        message = []
        for line in dialog.split("\n"):
            if line.startswith("user:"):
                message.append({"role": self.user, "content": line[5:]})
            elif line.startswith("robot:"):
                message.append({"role": self.robot, "content": line[6:]})
            # else:
            # raise ImportError(f"Could not find right role in {dialog}")
        return message

    def generate_tokens(self, messages):
        """_summary_
            prompt example:
            messages = [
                {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                {"role": "user", "content": "Who are you?"},
            ]
        Args:
            messages (_type_): _description_

        Returns:
            _type_: _description_
        """

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        return input_ids

    def show_input_tokens(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return input_ids


class LLM_loader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.last_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["internlm2-chat-1_8b","Meta-Llama-3-8B-Instruct"],),
                "dialog": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "user:Hello!",
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 256,
                        "min": 100,
                        "max": 200000,
                        "display": "number",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "display": "number",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "display": "number",
                    },
                ),
                "quantize":(["None","int8","int4"],),
                "is_stream":("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "chat"

    # OUTPUT_NODE = False

    CATEGORY = "ComfyLLM"

    def chat(self, model_name, dialog, max_new_tokens, temperature, top_p,quantize,is_stream):
        if self.last_model != model_name:
            if self.last_model is not None:
                self.model.to("CPU")
                self.tokenizer.to("CPU")
                del self.model
                del self.tokenizer
            else:
                self.tokenizer, self.model = load_model(model_name)
                token_processer = Text2Token(
                    model_name=model_name,
                    tokenizer=self.tokenizer,
                )

        messages = token_processer.dialog2message(dialog)
        if len(messages) < 1:
            return ("No message",)
        tokens = token_processer.generate_tokens(messages)
        outputs = self.model.generate(
            tokens.to(self.model.device),
            max_new_tokens=max_new_tokens,
            eos_token_id=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids(token_processer.eot),
            ],
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0][tokens.shape[-1] :]
        return (f"robot:{self.tokenizer.decode(response, skip_special_tokens=True)}",)


NODE_CLASS_MAPPINGS = {
    "LLM_loader": LLM_loader,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_loader": "LLM_loader",
}

if __name__ == "__main__":
    model_name = "Meta-Llama-3-8B-Instruct"
    dialog = "user:hello"
    tokenizer, model = load_model(model_name)
    token_processer = Text2Token(
        model_name=model_name,
        tokenizer=tokenizer,
    )
    messages = token_processer.dialog2message(dialog)
    print(messages)
    tokens = token_processer.generate_tokens(messages)
    outputs = model.generate(
        tokens.to(model.device),
        max_new_tokens=256,
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(token_processer.eot),
        ],
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][tokens.shape[-1] :]
    print(tokenizer.decode(response, skip_special_tokens=True))
    print("del model")
    time.sleep(3)
