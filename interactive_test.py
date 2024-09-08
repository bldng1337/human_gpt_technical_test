from typing import Any, Dict, List
import gradio as gr
from llama_cpp import Llama



syspropmt=r"""
The User will make an inquiry to the assistant.
Fullfill the users inquiry.
The User will write a message with his closing thoughts and the keyword "<|endtile|>" if his inquiry is fullfilled.
The User will never have more than one inquiry in one conversation.
The User will never complete his own inquiry.
The User will never be a assistant.
The User keep his message short in one sentence.
All conversations will end with "<|endtile|>".
After each User message is one assistant response.
There can never be more than one assistant response in succession.

Example:
User: What is the capital?
Assistant: Could you please specify which capital you are referring to?
User: The capital of France
Assistant: The capital of France is Paris
User: <|endtile|>
""".strip()

def chatmsg(message, role):
    return {"role": role, "content": message}

class Model:
    def __init__(self):
        pass
    def __call__(self, msg:str, stop:List[str], max_tokens:int):
        raise NotImplementedError
    def conv(self, msgs:List[Dict[str, str]]):
        raise NotImplementedError
    def starttok(self, user:str):
        raise NotImplementedError
    def close(self):
        pass

class Phi35RPMax(Model):
    def __init__(self):
        self.llm = Llama.from_pretrained(
            repo_id="ArliAI/Phi-3.5-mini-3.8B-ArliAI-RPMax-v1.1-GGUF",
            filename="ArliAI-RPMax-3.8B-v1.1-fp16.gguf",
        )
        
    def __call__(self, msg:str, stop:List[str], max_tokens:int):
        return self.llm(msg, stop=stop, max_tokens=max_tokens)
    
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
            filename="Phi-3.5-mini-instruct-IQ3_XS.gguf",
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
        )
    def __call__(self, msg:str, stop:List[str], max_tokens:int):
        return self.llm(msg, stop=stop, max_tokens=max_tokens)
    
    def conv(self,msgs:List[Dict[str, str]]):
        return "\n".join([f"<|begin_of_text|><|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>" for msg in msgs])
    def starttok(self,user:str):
        return f"<|begin_of_text|><|start_header_id|>{user}<|end_header_id|>\n\n"
    def close(self):
        self.llm.close()
Llama31uncensored.modelname="Llama31-uncensored-fp16"

class Llama31(Model):
    def __init__(self):
        self.llm = Llama.from_pretrained(
            repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
            filename="Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf",
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
currmodel=Phi35()
conversations:List[Dict[str, Any]]=[
    #More Trivia Style Question
    {"name":"Country","content":[{"role":"user","content":"What is the capital?"}]},
    {"name":"Simple Math","content":[{"role":"user","content":"What is 3*4?"}]},
    {"name":"Geography","content":[{"role":"user","content":"What is the capital of France?"}]},
    {"name":"History","content":[{"role":"user","content":"Who was the first president of the United States?"}]},
    {"name":"Science","content":[{"role":"user","content":"What is the chemical symbol for gold?"}]},
    {"name":"Literature","content":[{"role":"user","content":"Who wrote 'Romeo and Juliet'?"}]},
    {"name":"Technology","content":[{"role":"user","content":"What does 'CPU' stand for?"}]},
    {"name":"Sports","content":[{"role":"user","content":"In which sport would you use a shuttlecock?"}]},
    {"name":"Music","content":[{"role":"user","content":"Which instrument has 88 keys?"}]},
    {"name":"Food","content":[{"role":"user","content":"What is the main ingredient in guacamole?"}]},
    {"name":"Animals","content":[{"role":"user","content":"How many legs does a spider typically have?"}]},
    {"name":"Language","content":[{"role":"user","content":"What language is 'Bonjour' from?"}]},
    {"name":"Movies","content":[{"role":"user","content":"Who directed the movie 'Jurassic Park'?"}]},
    {"name":"Art","content":[{"role":"user","content":"Who painted the Mona Lisa?"}]},
    {"name":"Human Body","content":[{"role":"user","content":"How many chambers does the human heart have?"}]},
    {"name":"Astronomy","content":[{"role":"user","content":"What is the closest planet to the Sun?"}]},
    #Open Ended Question
    {"name":"Career Advice","content":[{"role":"user","content":"I'm considering a career change. Can you help me explore some options based on my interests in technology and creativity?"}]},
    {"name":"Travel Planning","content":[{"role":"user","content":"I want to plan a backpacking trip through Europe. What should I consider when planning my itinerary?"}]},
    {"name":"Environmental Issues","content":[{"role":"user","content":"What are some practical steps I can take to reduce my carbon footprint in daily life?"}]},
    {"name":"Personal Finance","content":[{"role":"user","content":"I'm new to investing. Can you explain some basic strategies for beginners?"}]},
    {"name":"Health and Wellness","content":[{"role":"user","content":"I'm looking to improve my overall health. What lifestyle changes would you recommend?"}]},
    {"name":"Technology Trends","content":[{"role":"user","content":"How might artificial intelligence impact job markets in the next decade?"}]},
    {"name":"Creative Writing","content":[{"role":"user","content":"I want to start writing a novel. Can you help me brainstorm some ideas for a science fiction story?"}]},
    {"name":"Ethical Dilemmas","content":[{"role":"user","content":"What are your thoughts on the ethical implications of genetic engineering in humans?"}]},
    {"name":"Cultural Understanding","content":[{"role":"user","content":"I'm moving to Japan for work. What cultural differences should I be aware of?"}]},
    {"name":"Future Technologies","content":[{"role":"user","content":"How do you think transportation might evolve in the next 50 years?"}]},
    {"name":"Personal Development","content":[{"role":"user","content":"I want to become more productive. What techniques or tools would you suggest?"}]},
    {"name":"Social Issues","content":[{"role":"user","content":"What are some effective ways to address income inequality in modern societies?"}]},
    {"name":"Education","content":[{"role":"user","content":"How might education systems need to change to better prepare students for the future job market?"}]},
    {"name":"Relationship Advice","content":[{"role":"user","content":"How can I improve communication in my relationships?"}]},
    {"name":"Philosophy","content":[{"role":"user","content":"What does it mean to live a good life in today's world?"}]},
    #Misc LLM Usage Questions
    {"name":"Writing Assistance","content":[{"role":"user","content":"Can you help me write a professional email to decline a job offer politely?"}]},
    {"name":"Code Debugging","content":[{"role":"user","content":"I'm getting a 'TypeError: cannot read property 'length' of undefined' in my JavaScript code. How can I fix this?"}]},
    {"name":"Recipe Modification","content":[{"role":"user","content":"I have a chocolate cake recipe, but I'm lactose intolerant. How can I modify it to be dairy-free?"}]},
    {"name":"Language Translation","content":[{"role":"user","content":"How do you say 'I'm sorry for the inconvenience' in Spanish?"}]},
    {"name":"Tech Support","content":[{"role":"user","content":"My Wi-Fi keeps disconnecting. What troubleshooting steps should I try?"}]},
    {"name":"Data Analysis","content":[{"role":"user","content":"I have a CSV file with sales data. Can you help me create a Python script to calculate monthly totals?"}]},
    {"name":"Legal Advice","content":[{"role":"user","content":"What are the basic steps to register a trademark for my small business?"}]},
    {"name":"Product Comparison","content":[{"role":"user","content":"Can you compare the features of the latest iPhone and Samsung Galaxy models?"}]},
    {"name":"Travel Information","content":[{"role":"user","content":"What documents do I need to travel from Germany to Canada as a tourist?"}]},
    {"name":"Health Query","content":[{"role":"user","content":"What are some natural remedies for reducing high blood pressure?"}]},
    {"name":"Market Research","content":[{"role":"user","content":"What are the current trends in sustainable fashion?"}]},
    {"name":"Event Planning","content":[{"role":"user","content":"I'm organizing a virtual team-building event. Can you suggest some fun online activities?"}]},
    {"name":"Academic Research","content":[{"role":"user","content":"Can you summarize the main arguments in Hardin's 'Tragedy of the Commons'?"}]},
    {"name":"DIY Project","content":[{"role":"user","content":"How do I build a raised garden bed? Can you give me step-by-step instructions?"}]},
    {"name":"Financial Advice","content":[{"role":"user","content":"What are the pros and cons of investing in index funds versus individual stocks?"}]}
]

with gr.Blocks() as demo:
    with gr.Accordion("Info"):
        gr.Markdown(f"""
                    # HumanGPT Game Test
                    ## Disclaimer
                    This is a test of feasibility and to evaluate different models, prompts, and types of conversations.
                    The current conversations don't represent the type of interactions the final game would have, but rather showcase various possible scenarios for playtesting and assessing model behavior.
                    This playground will also be used to test fine-tuned models in the future.
                    ## How to Use
                    - Use the chat window to test the model interactively.
                    - If the model responds with "End of conversation," it means the interaction is over.
                    - Change the conversation by selecting a different option from the choice box.
                    - Change the model by selecting a different option from the model choice box.
                    - To modify the system prompt, edit the text in the system prompt text box.
                    - If you choose Custom in the conversation choice box, you can enter a custom conversation in the text box under the Custom Conversation accordion.
                    """)
    chatbot = gr.Chatbot([chatmsg("What is the capital?","user")],type="messages",show_copy_all_button=True)
    msg = gr.Textbox()
    submit = gr.Button("Submit")
    with gr.Accordion("Config"):
        convchoicebox = gr.Radio(choices=[conversation["name"] for conversation in conversations]+["Custom"], value="Country", label="Conversations")
        with gr.Accordion("Custom Conversation",open=False):
            custom_conv=gr.Textbox(value="", label="Conversation")
            def update_custom_conv(custom_conv,convchoicebox,chatbot,msg):
                if(convchoicebox=="Custom"):
                    return "", [chatmsg(custom_conv,"user")]
                return msg,chatbot
            custom_conv.change(update_custom_conv, [custom_conv,convchoicebox,chatbot,msg], [msg,chatbot])
        def update_choicebox(choice,custom_conv):
            if(choice=="Custom"):
                return "", [chatmsg(custom_conv,"user")]
            return "", next(conversation for conversation in conversations if conversation["name"] == choice)["content"]
        sysprompt=gr.Textbox(value=syspropmt, label="System Prompt")
        convchoicebox.change(update_choicebox, [convchoicebox,custom_conv], [msg,chatbot])
        modelchoicebox = gr.Radio(choices=[model.modelname for model in models], value=currmodel.modelname, label="Model")
        def update_modelchoicebox(choice):
            global currmodel
            currmodel.close()
            currmodel=next(model for model in models if model.modelname == choice)()
            return "", []
        modelchoicebox.change(update_modelchoicebox, [modelchoicebox], [msg,chatbot])

    def respond(message:str, chat_history:List[Dict[str, str]],syspropmt:str):
        global currmodel
        if "End of conversation." in [i["content"] for i in chat_history]:
            return "", chat_history
        chat_history.append(chatmsg(message,"assistant"))
        ret=currmodel(currmodel.conv([chatmsg(syspropmt,"system")])+currmodel.conv(chat_history)+"<|user|>\n", stop=[".","\n \n","?\n",".\n","tile|>"],max_tokens=100)
        comp=ret["choices"][0]["text"]
        print(repr(comp))
        if("<|end" in comp):
            chat_history.append(chatmsg(comp.removesuffix("<|end"),"user"))
            chat_history.append(chatmsg("End of conversation.","user"))
        else:
            chat_history.append(chatmsg(comp,"user"))
        return "", chat_history
    submit.click(respond, [msg, chatbot,sysprompt], [msg, chatbot])
    msg.submit(respond, [msg, chatbot,sysprompt], [msg, chatbot])

demo.launch()