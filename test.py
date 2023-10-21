from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
import os
from os.path import dirname, join 
from dotenv import load_dotenv
from vecdb import cossimhist
#W6hkUqzTQWcL6IVw

#.env adjustments
dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)
API_TOKEN = os.environ.get("APIKEY")
PROJECT_ID = os.environ.get("PROJECT_ID")

#1) empty vec db
#2) append the first prompt + response {str: vector}
#3) before each next prompt + response is sent to the model, we run cosine similarity between the prompt and the database                    #500
#4) pick as many similarities as possible before filling out the context window OR hardcode a restriction in prompt length (limit the user to n chars, the rest of the context window -- to be used for history)
#5) after we pick the n top prompts + responses, pack them up into a prompt that says 'your last response to this prompt: prompt were such and such: response'
#6) the model answers

#chatbot class
class ChatbotWithHistory:
    def __init__(self):
        # self.prompt = 'Act as a helpful virtual psychology assistant that provides emotional support and friendly advice to children in a critical situation, situation of stress and trouble. You should provide the user with correct, guiding information. Upon receiving a prompt reply to the user immediately, without augmenting their prompt. DO NOT model a dialogue -- give them time to in turn reply to you. Assume that the prompt the user inputs is complete and there is no need for you to add anything to it. ===PROMPT START=== {userinput} ===PROMPT END===\n'
        # self.template = 'Act as a helpful virtual psychology assistant that provides emotional support and friendly advice to children in a critical situation, situation of stress and trouble. You should provide the user with correct, guiding information. Upon receiving a prompt reply to the user immediately, without augmenting their prompt. DO NOT model a dialogue -- give them time to in turn reply to you. Assume that the prompt the user inputs is complete and there is no need for you to add anything to it. ===PROMPT START=== {userinput} ===PROMPT END==='
        
        self.template = """Act as a helpful virtual psychology assistant that provides emotional support and friendly advice to children in a critical situation, situation of stress and trouble. You should provide the user with correct, guiding information. Upon receiving a prompt reply to the user immediately, without augmenting their prompt. DO NOT model a dialogue -- give them time to in turn reply to you. Assume that the prompt the user inputs is complete and there is no need for you to add anything to it.
        {chat_history}
        Human: {human_input}
        Chatbot:"""

        self.prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"], template=self.template
        )

        GenParams().get_example_values()

        #model hyperparameters 
        self.generate_params = {
            GenParams.MIN_NEW_TOKENS: 10,
            GenParams.MAX_NEW_TOKENS: 250,
            GenParams.TEMPERATURE: 0.0,
            GenParams.REPETITION_PENALTY: 1,
        }

        #initializing the model 
        self.model = Model(
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params=self.generate_params,
            credentials={
                "apikey": API_TOKEN,  
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id=PROJECT_ID
        )
        self.chain = ConversationChain(llm=self.model.to_langchain(), prompt=self.prompt, verbose=True, memory=self.memory)
        inp = {
    "new_prompt" : {
        "prompt" : str,
        "vectorized_prompt" : list(int(str)) #embedded str
    },
    "history" : [
        {
            "prompt" : str,
            "vectorized_prompt" : list(int(str)), #embedded str
            "answer" : str
        },
        {
            "prompt" : str,
            "vectorized_prompt" : list(int(str)), #embedded str
            "answer" : str
        },
        ...
    ]
}

    #a function to get the model's response to some prompt + history 
    def get_response(self, user_input:str, inp: dict):
        last_prompt = inp['new_prompt']['prompt'] #str of the last prompt
        
        #running cosine similarity on the whole  

        bot_response = self.llm_chain(human_input=prompt)
        return bot_response

#executing the file 
if __name__ == "__main__":
    chatbot = ChatbotWithHistory()

    response = chatbot('hey, my name is el. let`s talk')
    print(response)