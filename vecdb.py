import pinecone 
import openai
import os
#comweof
#the database class
class Database:
    def __init__(self):
        #initializing a pinecone vector db
        pinecone.init(      
            api_key='9386941d-63fa-4c52-8e4c-c829592e1a0c',      
            environment='gcp-starter')   
        
        self.index = pinecone.Index('conversation')
        self.openai.api_key = os.getenv("OPENAI_API")

    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        response = openai.Embedding.create(
            input = [text], 
            model=model)
        embedding = response['data'][0]['embedding']

        return embedding 
    