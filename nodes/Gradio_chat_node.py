import json
import urllib
import requests
import os
import copy

import subprocess

from multiprocessing import Process


root_dir = os.environ.get("ComLLM_Path")
print("*****", root_dir)


def parse_nodeid(nodeid, prompt):
    nodes_for_gradio = {}
    nodes = prompt["prompt"][2]
    for id in nodeid:
        node = copy.deepcopy(nodes[str(id)])
        node["inputs"] = {
            k: v for k, v in node["inputs"].items() if not isinstance(v, list)
        }
        nodes_for_gradio.update({str(id): node})
    return nodes_for_gradio


def find_node_output(prompt):
    nodes = prompt["prompt"][4]
    return nodes


def find_node_in_groups(prompt, cross_ratio=0.5):
    groups = prompt["prompt"][3]["extra_pnginfo"]["workflow"]["groups"]
    nodes_id_shown = []
    for group in groups:
        group_bound = group["bounding"]
        for node in prompt["prompt"][3]["extra_pnginfo"]["workflow"]["nodes"]:
            node_pos = node["pos"]
            node_size = node["size"]
            if isinstance(node_size, dict):
                node_size = [v for k, v in node["size"].items()]
            cross_x1 = max(group_bound[0], node_pos[0])
            cross_y1 = max(group_bound[1], node_pos[1])
            cross_x2 = min(group_bound[0] + group_bound[2], node_pos[0] + node_size[0])
            cross_y2 = min(group_bound[1] + group_bound[3], node_pos[1] + node_size[1])
            if cross_x2 > cross_x1 and cross_y2 > cross_y1:
                cross_area = (cross_x2 - cross_x1) * (cross_y2 - cross_y1)
                if cross_area > cross_ratio * node_size[0] * node_size[1]:
                    nodes_id_shown.append(node["id"])
    return nodes_id_shown


def send_prompt_requests(prompt):
    url = "http://127.0.0.1:8188/prompt"
    req = requests.post(url, data=json.dumps(prompt).encode("utf-8"))
    # resp=json.loads(urllib.request.urlopen(req).read())
    return req


def get_history():
    """
    Get the history of the current session.
    """
    utl = "http://127.0.0.1:8188/history"
    req = urllib.request.Request(utl)
    resp = urllib.request.urlopen(req).read().decode("utf-8")
    resp = json.loads(resp)
    return resp


class Gradio_chat:
    def __init__(self):
        self.clinet_on = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "user:Hello!",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "show"

    # OUTPUT_NODE = False

    CATEGORY = "ComfyLLM/Gradio"

    def show(self, text):
        print('****** run clinet sp1')
        if not self.clinet_on:
            dir_path=os.path.dirname(os.path.abspath(__file__))
            gradio_clinet_path=os.path.join(dir_path,'gradio_clinet.py')
            process = subprocess.Popen(f'python {gradio_clinet_path}', shell=True)
            print('****** run clinet sp2')

        return (f"Success",)


class Chat_storage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dialog": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "user:Hello!",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "chat"

    # OUTPUT_NODE = False

    CATEGORY = "ComfyLLM"

    def chat(self, dialog):
        return (dialog,)


NODE_CLASS_MAPPINGS = {"Gradio_chat": Gradio_chat, "Chat_storage": Chat_storage}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gradio_chat": "Gradio_chat",
    "Chat_storage": "Chat_storage",
}
