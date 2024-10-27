from typing import Dict, List

from llama_cpp import Llama
llama_args={"n_gpu_layers":100,"main_gpu":0,"verbose":True}

class Model:
    def __init__(self):
        pass
    def __call__(self, msg:str, stop:List[str], max_tokens:int):
        raise NotImplementedError
    def conv(self, msgs:List[Dict[str, str]])->str:
        raise NotImplementedError
    def starttok(self, user:str)->str:
        raise NotImplementedError
    def start(self)->str:
        return ""
    def close(self):
        pass

class Phi35RPMax(Model):
    def __init__(self):
        self.llm = Llama.from_pretrained(
            repo_id="ArliAI/Phi-3.5-mini-3.8B-ArliAI-RPMax-v1.1-GGUF",
            filename="ArliAI-RPMax-3.8B-v1.1-fp16.gguf",
            **llama_args,

        )
        
    def __call__(self, msg:str, stop:List[str], max_tokens:int):
        ret=self.llm(msg, stop=stop, max_tokens=max_tokens)
        return ret
    
    def conv(self,msgs:List[Dict[str, str]]):
        return "\n".join([f"<|{msg['role']}|>\n{msg['content']}<|end|>" for msg in msgs])
    def starttok(self,user:str):
        return f"<|{user}|>\n"
    def close(self):
        self.llm.close()
Phi35RPMax.modelname="Phi35RPMax-fp16"
class Phi35(Model):
    def __init__(self):
        self.llm = Llama.from_pretrained(
            repo_id="bartowski/Phi-3.5-mini-instruct-GGUF",
            filename="Phi-3.5-mini-instruct-f32.gguf",
            **llama_args,
        )
    def __call__(self, msg:str, stop:List[str], max_tokens:int):
        return self.llm(msg, stop=stop, max_tokens=max_tokens)

    def conv(self,msgs:List[Dict[str, str]]):
        return "\n".join([f"<|{msg['role']}|>\n{msg['content']}<|end|>" for msg in msgs])
    
    def starttok(self,user:str):
        return f"<|{user}|>\n"
    def close(self):
        self.llm.close()
Phi35.modelname="Phi35-IQ3_XS"

# TODO: Gemma2 needs license maybe try it in the future but dont think it is worth it
# class Gemma2(Model):
#     def __init__(self):
#         self.llm = Llama.from_pretrained(
#             repo_id="google/gemma-2-2b-it-GGUF",
#             filename="2b_it_v2.gguf",
#         )
#     def __call__(self, msg:str, stop:List[str], max_tokens:int):
#         return self.llm(msg, stop=stop, max_tokens=max_tokens)
    
#     def conv(self,msgs:List[Dict[str, str]]):#https://ai.google.dev/gemma/docs/formatting?hl=de
#         return "\n".join([f"<|{msg['role']}|>\n{msg['content']}<|end|>" for msg in msgs])
#     def formatmessage(self,msg:str, role:str):#https://ai.google.dev/gemma/docs/formatting?hl=de
#         if(role=="system"):
#             # Gemma2 does not support system messages / isnt trained for them
#             # TODO: Make them Assistant messages and test if this improves the results
#             return ""
#         if role=="assistant":
#             role="model"
#         return f"<start_of_turn>{role}\n{msg}<end_of_turn>"
#     def starttok(self,user:str):
#         return f"<start_of_turn>{user}\n"
#     def close(self):
#         self.llm.close()
# Gemma2.modelname="Gemma2-2b-it-GGUF"

class Llama31uncensored(Model):
    def __init__(self):
        self.llm = Llama.from_pretrained(
            repo_id="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
            filename="Llama-3.1-8B-Lexi-Uncensored_V2_F16.gguf",
            **llama_args,
        )
    def __call__(self, msg:str, stop:List[str], max_tokens:int):
        return self.llm(msg, stop=stop, max_tokens=max_tokens)
    
    def start(self):
        return "<|begin_of_text|>"
    
    def conv(self,msgs:List[Dict[str, str]]):
        return "\n".join([f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>" for msg in msgs])
    def starttok(self,user:str):
        return f"<|start_header_id|>{user}<|end_header_id|>\n\n"
    def close(self):
        self.llm.close()
Llama31uncensored.modelname="Llama31-uncensored-fp16"

class Llama31(Model):
    def __init__(self):
        self.llm = Llama.from_pretrained(
            repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf",
            **llama_args,
        )
    def __call__(self, msg:str, stop:List[str], max_tokens:int):
        return self.llm(msg, stop=stop, max_tokens=max_tokens)
    
    def conv(self,msgs:List[Dict[str, str]]):
        return "\n".join([f"<|begin_of_text|><|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>" for msg in msgs])
    def starttok(self,user:str):
        return f"<|begin_of_text|><|start_header_id|>{user}<|end_header_id|>"
    def close(self):
        self.llm.close()
Llama31.modelname="Llama31-IQ4_XS"

models=[Phi35RPMax,Phi35,Llama31uncensored,Llama31]