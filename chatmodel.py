from typing import Dict, List, Literal, TypedDict


from models import Model
from pybars import Compiler
compiler = Compiler()

class Turn(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str

def chatmsg(message:str, role:Literal["user", "assistant", "system"]):
    return {"role": role, "content": message}

conversation=List[Turn]

class ChatModel:
    def __init__(self,model:Model,sysprompt:str):
        self.setModel(model)
        self.setSysPrompt(sysprompt)
    def __call__(self, msg:str):
        raise NotImplementedError
    def getconversation(self) -> conversation:
        raise NotImplementedError
    def conversationend(self) -> bool:
        raise NotImplementedError
    def setconversation(self,conversation:conversation):
        raise NotImplementedError
    def setSysPrompt(self,sysprompt:str):
        def _eq(this, a,b):
            return a==b
        self.sysprompt=compiler.compile(sysprompt)({
            "model":self.name
        },helpers={"eq":_eq})
        print(self.name+" SystemPrompt:\n"+self.sysprompt)
    def setModel(self,model:Model):
        self.model=model

class SwapChatModel(ChatModel):
    def __init__(self,model:Model,sysprompt:str):
        super().__init__(model,sysprompt)
        self.conversation=[]
    def __call__(self, msg:str):
        if "End of conversation." in [i["content"] for i in self.conversation]:
            return
        self.conversation.append(chatmsg(msg,"assistant"))
        prompt="".join([
            self.model.start(),
            self.model.conv([chatmsg(self.sysprompt,"system")]),
            self.model.conv(self.conversation),self.model.starttok("user")
            ])
        ret=self.model(prompt, stop=[".","\n \n","?\n",".\n","tile|>","\n"],max_tokens=100)
        comp=ret["choices"][0]["text"]
        if("<|end" in comp):
            self.conversation.append(chatmsg(comp.removesuffix("<|end"),"user"))
            self.conversation.append(chatmsg("End of conversation.","user"))
        else:
            self.conversation.append(chatmsg(comp,"user"))
    def getconversation(self) -> conversation:
        return self.conversation
    def conversationend(self) -> bool:
        return "End of conversation." in [i["content"] for i in self.conversation]
    def setconversation(self,conversation:conversation):
        self.conversation=conversation
SwapChatModel.name="SwapChat"


class InquiryChatModel(SwapChatModel):
    def __init__(self,model:Model,sysprompt:str):
        super().__init__(model,sysprompt)
    def inquire(self,msg):
        prompt="".join([
            self.model.start(),
            self.model.conv([chatmsg(self.sysprompt,"system")]),
            self.model.conv(self.conversation),
            self.model.conv([chatmsg(msg,"assistant")]),
            self.model.starttok("system"),
            "Is this conversation complete(true/false)?\n"
            ])
        ret=self.model(prompt, stop=[".","\n \n","?\n",".\n","tile|>","\n"],max_tokens=10)
        print("system prompt:",ret["choices"][0]["text"])
        if "true" in ret["choices"][0]["text"].lower():
            self.conversation.append(chatmsg(msg,"user"))
            self.conversation.append(chatmsg("End of conversation.","user"))
    def __call__(self, msg:str):
        self.inquire(msg)
        super().__call__(msg)
InquiryChatModel.name="InquiryChat"
models=[SwapChatModel,InquiryChatModel]