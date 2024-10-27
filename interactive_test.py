from typing import Any, Dict, List
import gradio as gr
from llama_cpp import Llama

import chatmodel
from models import Phi35,models

sysprompt=r"""
{{! This comment will not show up in the output}}
The User will make an inquiry to the assistant.
Fullfill the users inquiry.
{{#if (eq model "SwapChat")}}
The User will write a message with his closing thoughts and the keyword "<|endtile|>" if his inquiry is fulfilled.
{{/if}}
The User will never have more than one inquiry in one conversation.
The User will never complete his own inquiry.
The User will never be a assistant.
The User keep his message short in one sentence.
{{#if (eq model "SwapChat")}}
All conversations will end with "<|endtile|>".
{{/if}}
After each User message is one assistant response.
There can never be more than one assistant response in succession.
{{#if (eq model "SwapChat")}}
Example:
User: What is the capital?
Assistant: Could you please specify which capital you are referring to?
User: The capital of France
Assistant: The capital of France is Paris
User: <|endtile|>
{{else}}
Example:
User: What is the capital?
Assistant: Could you please specify which capital you are referring to?
User: The capital of France
Assistant: The capital of France is Paris
{{/if}}
""".strip()

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

def chatmsg(message, role):
    return {"role": role, "content": message}



currmodel=Phi35()
chat:chatmodel.ChatModel=chatmodel.models[0](currmodel,sysprompt)

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
    chatbot = gr.Chatbot(conversations[0]["content"],type="messages",show_copy_all_button=True)
    msg = gr.Textbox()
    submit = gr.Button("Submit")
    with gr.Accordion("Config"):
        #Choose Conversations
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
        convchoicebox.change(update_choicebox, [convchoicebox,custom_conv], [msg,chatbot])

        msysprompt=gr.Textbox(value=sysprompt, label="System Prompt")

        def update_sysprompt(csysprompt:str):
            global sysprompt
            sysprompt=csysprompt
            chat.setSysPrompt(sysprompt)
            chat.setconversation([])
            return "", chat.getconversation()
        msysprompt.submit(update_sysprompt, [msysprompt], [msg,chatbot])

        #Choose Models
        modelchoicebox = gr.Radio(choices=[model.modelname for model in models], value=currmodel.modelname, label="Model")
        def update_modelchoicebox(choice):
            global currmodel
            currmodel.close()
            currmodel=next(model for model in models if model.modelname == choice)()
            chat.setModel(currmodel)
            chat.setconversation([])
            return "", chat.getconversation()
        modelchoicebox.change(update_modelchoicebox, [modelchoicebox], [msg,chatbot])

        chatchoicebox = gr.Radio(choices=[model.name for model in chatmodel.models], value=chat.name, label="Chat")
        def update_chatchoicebox(choice):
            global chat, currmodel, sysprompt
            chat=next(model for model in chatmodel.models if model.name == choice)(currmodel,sysprompt)
            chat.setconversation([])
            return "", chat.getconversation()
        chatchoicebox.change(update_chatchoicebox, [chatchoicebox], [msg,chatbot])

    #generate response
    def respond(message:str,chatbot:List[Dict[str, str]]):
        global chat
        chat.setconversation(chatbot)
        chat(message)
        return "", chat.getconversation()
    submit.click(respond, [msg,chatbot], [msg, chatbot])
    msg.submit(respond, [msg,chatbot], [msg, chatbot])
demo.launch()
