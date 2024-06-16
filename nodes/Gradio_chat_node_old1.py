import json
import gradio as gr
import urllib
import requests
import websocket
import uuid
import os
import copy
import httpx
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1089'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1089'

from PIL import Image
import numpy as np
from multiprocessing import Process


# root_dir = os.environ.get("ComLLM_Path")


# def parse_nodeid(nodeid, prompt):
#     nodes_for_gradio = {}
#     nodes = prompt["prompt"][2]
#     for id in nodeid:
#         node = copy.deepcopy(nodes[str(id)])
#         node["inputs"] = {
#             k: v for k, v in node["inputs"].items() if not isinstance(v, list)
#         }
#         nodes_for_gradio.update({str(id): node})
#     return nodes_for_gradio


# def find_node_output(prompt):
#     nodes = prompt["prompt"][4]
#     return nodes


# def find_node_in_groups(prompt, cross_ratio=0.5):
#     groups = prompt["prompt"][3]["extra_pnginfo"]["workflow"]["groups"]
#     nodes_id_shown = []
#     for group in groups:
#         group_bound = group["bounding"]
#         for node in prompt["prompt"][3]["extra_pnginfo"]["workflow"]["nodes"]:
#             node_pos = node["pos"]
#             node_size = node["size"]
#             if isinstance(node_size, dict):
#                 node_size = [v for k, v in node["size"].items()]
#             cross_x1 = max(group_bound[0], node_pos[0])
#             cross_y1 = max(group_bound[1], node_pos[1])
#             cross_x2 = min(group_bound[0] + group_bound[2], node_pos[0] + node_size[0])
#             cross_y2 = min(group_bound[1] + group_bound[3], node_pos[1] + node_size[1])
#             if cross_x2 > cross_x1 and cross_y2 > cross_y1:
#                 cross_area = (cross_x2 - cross_x1) * (cross_y2 - cross_y1)
#                 if cross_area > cross_ratio * node_size[0] * node_size[1]:
#                     nodes_id_shown.append(node["id"])
#     return nodes_id_shown


# def send_prompt_requests(prompt):
#     url = "http://127.0.0.1:8188/prompt"
#     req = requests.post(url, data=json.dumps(prompt).encode("utf-8"))
#     # resp=json.loads(urllib.request.urlopen(req).read())
#     return req


# def get_history():
#     """
#     Get the history of the current session.
#     """
#     utl = "http://127.0.0.1:8188/history"
#     req = urllib.request.Request(utl)
#     resp = urllib.request.urlopen(req).read().decode("utf-8")
#     resp = json.loads(resp)
#     return resp


# class Gradio_chat:
#     def __init__(self):
#         self.clinet_on=False

#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "text": (
#                     "STRING",
#                     {
#                         "multiline": True,
#                         "default": "user:Hello!",
#                     },
#                 ),
#             },
#         }

#     RETURN_TYPES = ("STRING",)
#     # RETURN_NAMES = ("image_output_name",)

#     FUNCTION = "show"

#     # OUTPUT_NODE = False

#     CATEGORY = "ComfyLLM/Gradio"
    
#     # def get_storage(self,):
#     #     prompt_history = get_history()
#     #     latest_prompt = list(prompt_history.values())[-1]
#     #     return list(latest_prompt['outputs'].values())[-1][]
        

#     def predict(self, message, message_history):
#         client_id = str(uuid.uuid4())
#         ws = websocket.WebSocket()
#         ws.connect("ws://127.0.0.1:8188/ws?clientId={}".format(client_id))
#         resp = send_prompt_requests({"prompt": self.prompt})

#         while True:
#             out = ws.recv()
#             print(out)
#             if isinstance(out, str):
#                 message = json.loads(out)
#                 if message["type"] == "executing":
#                     data = message["data"]
#                     if data["prompt_id"] == resp.json()["prompt_id"]:
#                         if data["node"] is None:
#                             break

#         message = message + json.dumps(self.input_nodes)
#         for i in range(len(message)):
#             # time.sleep(0.05)
#             yield "You typed: " + message[: i + 1]
#         return ["hello", "123"]

#     def run_clinet(self,):
#         self.clinet_on=True
#         prompt_history = get_history()
#         latest_prompt = list(prompt_history.values())[-1]

#         self.prompt = latest_prompt["prompt"][2]
#         self.node_in_group = find_node_in_groups(self.prompt)
#         self.input_nodes = parse_nodeid(self.node_in_group, self.prompt)
#         self.node_output = find_node_output(self.prompt)
#         self.output_nodes = parse_nodeid(self.node_output, self.prompt)
        
        

#         with gr.Blocks() as demo:
#             with gr.Row():
#                 with gr.Column(scale=1):
#                     for nodeid, node in self.prompt.items():
#                         with gr.Column():
#                             inputs = []

#                             def update_node(*args, **kwargs):
#                                 for i, (k, v) in enumerate(
#                                     self.input_nodes[nodeid]["inputs"].items()
#                                 ):
#                                     self.input_nodes[nodeid]["inputs"][k] = args[i]
#                                 self.prompt.update(self.input_nodes)

#                             node_copy = copy.deepcopy(node)
#                             for key, value in node_copy["inputs"].items():
#                                 if isinstance(value, int):
#                                     inputs.append(gr.Number(label=key, value=value))
#                                 elif isinstance(value, float):
#                                     inputs.append(gr.Number(label=key, value=value))
#                                 elif isinstance(value, str):
#                                     inputs.append(gr.Textbox(label=key, value=value))
#                             gr.Interface(
#                                 fn=update_node,
#                                 inputs=inputs,
#                                 outputs=None,
#                                 description=node_copy["class_type"],
#                             )
#                 with gr.Column(scale=3):
#                     btn1 = gr.ChatInterface(self.predict)
#         demo.launch()
        
#     def show(self, text):
#         if not self.clinet_on:
#             p = Process(target=self.run_clinet)
#             p.start()
            
#         return (f"Success",)

# class Chat_storage:
#     def __init__(self):
#         pass
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "dialog": (
#                     "STRING",
#                     {
#                         "multiline": True,
#                         "default": "user:Hello!",
#                     },
#                 ),
#             },
#         }

#     RETURN_TYPES = ("STRING",)
#     # RETURN_NAMES = ("image_output_name",)

#     FUNCTION = "chat"

#     # OUTPUT_NODE = False

#     CATEGORY = "ComfyLLM"

#     def chat(self, dialog):
#         return (dialog,)

# NODE_CLASS_MAPPINGS = {
#     "Gradio_chat": Gradio_chat,
#     "Chat_storage":Chat_storage
#     }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Gradio_chat": "Gradio_chat",
#     "Chat_storage":"Chat_storage",
# }
