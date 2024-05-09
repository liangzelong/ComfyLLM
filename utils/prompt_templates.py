class init_template():
    def __init__(self,model_name):
        self.model_name = model_name
        getattr(self,f'{self.model_name}'.replace('-','_'))()
        
    def internlm2_chat_1_8b(self):
        self.user_prompt = "<|User|>:{user}\n"
        self.robot_prompt = "<|Bot|>:{robot}<eoa>\n"
        self.cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"
    
    def Meta_Llama_3_8B_Instruct(self):
        self.user_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>'
        self.robot_prompt = '<|start_header_id|>assistant<|end_header_id|>\n\n{robot}<|eot_id|>'
        self.cur_query_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

def combine_history(model_template,prompt,history=None,if_clear=False):
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
    if model_name=="internlm2-chat-1_8b":
        return "<|im_end|>"
    elif model_name=="Meta-Llama-3-8B-Instruct":
        return "<|eot_id|>"

class Text2Token():
    eot_map={
        "internlm2-chat-1_8b": "<|eot_id|>",
        "Meta-Llama-3-8B-Instruct": "<|eot_id|>",
    }
    role_map={
        "internlm2-chat-1_8b":{
            "user":"user",
            "robot":"assistant",
        },
        "Meta-Llama-3-8B-Instruct":{
            "user":"user",
            "robot":"system"
        }
    }
    def __init__(self,model_name,tokenizer):
        self.model_name=model_name
        self.tokenizer=tokenizer
        self.eot=self.eot_map.get(self.model_name)
        if self.eot is None:
            raise NotImplementedError(f'{self.model_name} is not implemented')
        self.user=self.role_map[self.model_name]["user"]
        self.robot=self.role_map[self.model_name]["robot"]
    
    def dialog2message(self,dialog):
        """_summary_
            dilog example
            user:hello!\nrobot:hello\n
        Args:
            dialog (_type_): _description_

        Returns:
            _type_: _description_
        """        
        message=[]
        for line in dialog.split('\n'):
            if line.startswith("user:"):
                message.append({"role":self.user,"content":line[5:]})
            elif line.startswith("robot:"):
                message.append({"role":self.robot,"content":line[6:]})
            # else:
                # raise ImportError(f"Could not find right role in {dialog}")
        return message
        
    def generate_tokens(self,messages):
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

        
        input_ids=self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
            )
        return input_ids
    
    def show_input_tokens(self,messages):
        input_ids=self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            )
        return input_ids
    
    
# print(repr(combine_history('Meta-Llama-3-8B-Instruct','an apple',[{'role':'user','content':'hello'},{'role':'robot','content':'what do you want?'}])))