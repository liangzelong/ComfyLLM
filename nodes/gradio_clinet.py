import json
import gradio as gr
import urllib
import requests
import websocket
import uuid
import os
import copy
import time

from PIL import Image
import numpy as np


root_dir = os.environ.get("ComLLM_Path")

def update_dict_recursive(original, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and key in original:
            update_dict_recursive(original[key], value)
        else:
            original[key] = value

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

        

def predict(self, message, message_history):
    len_history=len(get_history())
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect("ws://127.0.0.1:8188/ws?clientId={}".format(client_id))
    resp = send_prompt_requests({"prompt": last_prompt})
    while True:
        if len(get_history()) > len_history:
            break
        else:
            time.sleep(1)

    # while True:
    #     out = ws.recv()
    #     print(out)
    #     if isinstance(out, str):
    #         message = json.loads(out)
    #         if message["type"] == "status":
    #             data = message["data"]
                # if data["prompt_id"] == resp.json()["prompt_id"]:
                #     if data["node"] is None:
                #         break
                    
    message = message + json.dumps(input_nodes)
    for i in range(len(message)):
        # time.sleep(0.05)
        yield "You typed: " + message[: i + 1]
    return ["hello", "123"]



prompt_history = get_history()
len_prompt=len(prompt_history)
while True:
    try:
        latest_run = list(prompt_history.values())[len_prompt]
        last_prompt = latest_run["prompt"][2]
        node_in_group = find_node_in_groups(latest_run)
        input_nodes = parse_nodeid(node_in_group, latest_run)
        node_output = find_node_output(latest_run)
        output_nodes = parse_nodeid(node_output, latest_run)
        break
    except:
        len_prompt=len_prompt-1


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            for nodeid, node in input_nodes.items():
                with gr.Column():
                    inputs = []

                    def update_node(*args, **kwargs):
                        for i, (k, v) in enumerate(
                            input_nodes[nodeid]["inputs"].items()
                        ):
                            input_nodes[nodeid]["inputs"][k] = args[i]
                        update_dict_recursive(last_prompt, input_nodes)

                    node_copy = copy.deepcopy(node)
                    for key, value in node_copy["inputs"].items():
                        if isinstance(value, int):
                            inputs.append(gr.Number(label=key, value=value))
                        elif isinstance(value, float):
                            inputs.append(gr.Number(label=key, value=value))
                        elif isinstance(value, str):
                            inputs.append(gr.Textbox(label=key, value=value))
                    gr.Interface(
                        fn=update_node,
                        inputs=inputs,
                        outputs=None,
                        description=node_copy["class_type"],
                    )
        with gr.Column(scale=3):
            btn1 = gr.ChatInterface(predict)
demo.launch()
        
