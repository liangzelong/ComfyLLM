# import os
# import time

# import json
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,pipeline
# from .utils.prompt_templates import Text2Token

# root_dir=os.path.dirname(os.path.dirname(__file__))
# print(os.path.join(root_dir,'support_models.json'))
# with open(os.path.join(root_dir,'support_models.json'),"r") as f:
#     model_maps=json.load(f)["model_maps"]
# model_list=list(model_maps.keys())


# def load_model(model_name):
#     nf4_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         load_in_8bit=False,
#         llm_int8_threshold=6.0,
#         llm_int8_has_fp16_weight=False,
#         bnb_4bit_compute_dtype=torch.float16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         os.path.join(root_dir, model_maps[model_name]),
#         device_map="auto",
#         torch_dtype=torch.float16,
#         trust_remote_code=True,
#         quantization_config=nf4_config,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(
#         os.path.join(root_dir, model_maps[model_name]), trust_remote_code=True
#     )
#     return tokenizer, model


# class LLM_loader:
#     def __init__(self):
#         self.model = None
#         self.tokenizer = None
#         self.last_model=None

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "model_name": (model_list,),
#                 "dialog": (
#                     "STRING",
#                     {
#                         "multiline": True,  # True if you want the field to look like the one on the ClipTextEncode node
#                         "default": "user:Hello!",
#                     },
#                 ),
#                 "max_new_tokens": (
#                     "INT",
#                     {
#                         "default": 256,
#                         "min": 100,  # Minimum value
#                         "max": 200000,  # Maximum value
#                         "step": 1,  # Slider's step
#                         "display": "number",  # Cosmetic only: display as "number" or "slider"
#                     },
#                 ),
#                 "temperature": (
#                     "FLOAT",
#                     {
#                         "default": 0.6,
#                         "min": 0.0,
#                         "max": 1.0,
#                         "step": 0.01,
#                         "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
#                         "display": "number",
#                     },
#                 ),
#                 "top_p": (
#                     "FLOAT",
#                     {
#                         "default": 0.9,
#                         "min": 0.0,
#                         "max": 1.0,
#                         "step": 0.01,
#                         "round": 0.001,  # The value represeting the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
#                         "display": "number",
#                     },
#                 ),
#             },
#         }

#     RETURN_TYPES = ("STRING",)
#     # RETURN_NAMES = ("image_output_name",)

#     FUNCTION = "chat"

#     # OUTPUT_NODE = False

#     CATEGORY = "ComfyLLM"

#     def chat(
#         self, model_name, dialog, max_new_tokens, temperature, top_p
#     ):
#         if self.last_model!=model_name:
#             if self.last_model is not None:
#                 self.model.to('CPU')
#                 self.tokenizer.to('CPU')
#                 del self.model
#                 del self.tokenizer
#             else:
#                 self.tokenizer, self.model = load_model(model_name)
#                 token_processer = Text2Token(
#                         model_name=model_name,
#                         tokenizer=self.tokenizer,
#                     )
        
        
        
        
#         messages = token_processer.dialog2message(dialog)
#         if len(messages) < 1:
#             return ("No message",)
#         tokens = token_processer.generate_tokens(messages)
#         outputs = self.model.generate(
#             tokens.to(self.model.device),
#             max_new_tokens=max_new_tokens,
#             eos_token_id=[
#                 self.tokenizer.eos_token_id,
#                 self.tokenizer.convert_tokens_to_ids(token_processer.eot),
#             ],
#             do_sample=True,
#             temperature=temperature,
#             top_p=top_p,
#         )
#         response = outputs[0][tokens.shape[-1]:]
#         return (f"robot:{self.tokenizer.decode(response, skip_special_tokens=True)}",)


# NODE_CLASS_MAPPINGS = {
#     "LLM_loader": LLM_loader,
# }


NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_loader": "LLM_loader",
}


