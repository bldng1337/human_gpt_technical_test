from typing import Dict, List
import gradio as gr
from llama_cpp import Llama

syspropmt=r"""
<|system|>
The User will make an inquiry to the assistant.
Fullfill the users inquiry.
The User will write a message with the content "<|endtile|>" if his inquiry is fullfilled.
The User will never have more than one inquiry in one conversation.
The User will never complete his own inquiry.
All conversations will end with "<|endtile|>".
After each User message is one assistant response.
There can never be more than one assistant response in succession.
The end of each user and assistant message is marked with "<|end|>".


Example:
<|user|>
What is the capital?<|end|>
<|assistant|>
Could you please specify which capital you are referring to?<|end|>
<|user|>
The capital of France<|end|>
<|assistant|>
The capital of France is Paris
<|user|>
<|endtile|><|end|>
""".strip()

llm = Llama.from_pretrained(
	repo_id="bartowski/Phi-3.5-mini-instruct-GGUF",
	filename="Phi-3.5-mini-instruct-IQ3_XS.gguf",
    n_gpu_layers=-1,
)

def chatmsg(message, role):
    return {"role": role, "content": message}


def conv(msgs:List[Dict[str, str]]):
    return "\n".join([f"<|{msg['role']}|>\n{msg['content']}<|end|>" for msg in msgs])

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([chatmsg("What is the capital?","user")],type="messages")
    msg = gr.Textbox()
    submit = gr.Button("Submit")
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history:List[Dict[str, str]]):
        if "End of conversation." in [i["content"] for i in chat_history]:
            return "", chat_history
        chat_history.append(chatmsg(message,"assistant"))
        ret=llm(syspropmt+conv(chat_history)+"<|user|>\n", stop=["\n \n","?\n",".\n","tile|>"],max_tokens=100)#Only stop at tile|> to see if the conv got terminated
        comp=ret["choices"][0]["text"]
        print(repr(comp))
        if("<|end" in comp):
            chat_history.append(chatmsg("End of conversation.","user"))
        else:
            chat_history.append(chatmsg(comp,"user"))
        return "", chat_history

    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()