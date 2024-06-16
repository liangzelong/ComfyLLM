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
from flask import Flask, request, jsonify
from flask import Response, stream_with_context, json
from flask_cors import CORS
import time
from PIL import Image
import numpy as np
import io
import base64


app = Flask(__name__)
CORS(app)

root_dir = os.environ.get("ComLLM_Path")


class RAG_loader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.last_model = None
        self.idx = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["Llama3", "InternLM_chat1_8B"],),
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
                "quantize": (["None", "int8", "int4"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "chat"

    # OUTPUT_NODE = False

    CATEGORY = "ComfyLLM"

    def chat(self, model_name, dialog, max_new_tokens, temperature, top_p, quantize):
        def generate_data():
            data_chunks1 = []
            data_chunks2 = []
            res = [
                "分治（divide and conquer）是一种非常重要且常见的算法策略。它通常基于递归实现，包括“分”和“治”两个步骤。\n “分”阶段是指递归地将原问题分解为两个或多个子问题，直至到达最小子问题时终止。这个阶段将原问题分解成规模更小、类似的子问题。 \n  “治”阶段是指从已知解的最小子问题开始，从底至顶地将子问题的解进行合并，从而构建出原问题的解。在这个阶段，子问题的解可以合并，形成原问题的解。\n  因此，分治是一种将原问题分解成子问题，然后将子问题的解合并以解决原问题的算法策略。",
                "robot:分治（divide and conquer）是指一种非常重要且常见的算法策略。它通常基于递归实现，包括“分”和“治”两个步骤。\n “分”步骤是指递归地将原问题分解为两个或多个子问题，直至到达最小子问题时终止。这个步骤的目的是将原问题分解成规模更小、类似的子问题，以便更好地解决。\n “治”步骤是指从已知解的最小子问题开始，从底至顶地将子问题的解进行合并，从而构建出原问题的解。这个步骤的目的是将子问题的解合并成原问题的解。 \n 分治策略的优点是可以将复杂的问题分解成更小、更简单的问题，进而解决。同时，它也可以将问题的规模减小，从而提高解决问题的效率。\n 在实际应用中，分治策略可以用来解决许多问题，例如归并排序、快速傅里叶变换、快速 Fourier Transform（FFT）等。",
            ]
            i = len(res[0]) // 3 + 1
            for j in range(i):
                data_chunks1.append(
                    {
                        "model": "llama3",
                        "message": {
                            "role": "assistant",
                            "content": res[j * 3 : min(len(res), (j + 1) * 3)],
                            "images": None,
                        },
                        "done": False,
                    },
                )
            i = len(res[1]) // 3 + 1
            for j in range(i):
                data_chunks2.append(
                    {
                        "model": "llama3",
                        "message": {
                            "role": "assistant",
                            "content": res[j * 3 : min(len(res), (j + 1) * 3)],
                            "images": None,
                        },
                        "done": False,
                    },
                )
            data_chunks = [data_chunks1, data_chunks2]

            for chunk in data_chunks[self.idx - 1]:
                time.sleep(0.1)
                yield json.dumps(chunk) + "\n"

        @app.route("/api/chat", methods=["POST"])
        def api():
            data = request.get_json()
            if not data:
                return jsonify({"error": "No input data provided"}), 400
            print(data)
            self.idx = self.idx + 1
            print(self.idx)
            return Response(
                stream_with_context(generate_data()),
            )

        app.run(debug=True, port=11434)
        return (f"Success",)



NODE_CLASS_MAPPINGS = {
    "Chroma_upadate": RAG_loader,
    "RAG": RAG_loader,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "Chroma_upadate": "Chroma_upadate",
    "RAG": "RAG",
}
