import pinecone 
from llama_index.embeddings import HuggingFaceEmbedding
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os
from os.path import dirname, join 
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#.env adjustments
dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)
PINECONE_API = os.environ.get("PINECONE_API")
OPENAI_API = os.environ.get("OPENAI_API")


#the database class
class VectorDatabase:
    def __init__(self):
        #initializing a pinecone vector db
        pinecone.init(      
            api_key=PINECONE_API,      
            environment='gcp-starter')   
        
        self.index_name = 'conv'
        self.index = pinecone.Index(self.index_name)
        # self.embeddings = OpenAIEmbeddings(
        #     model="text-embedding-ada-002",
        #     openai_api_key=OPENAI_API
        # )

        # self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # self.embed_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API)
        
        self.embed_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    def add_to_db(self, ind: str):
         embedding = self._get_embedding(ind)
         self.index.upsert([
            (ind, embedding),
         ])

    def _get_embedding(self, text:str):
        # embedding = self.embed_model.embed_query(text)
        embedding = self.embed_model.embed_query(text) 
          

        return embedding
    
    def get_k_vectors(self, vec: list):
        response = self.index.query(vector=vec,
                                top_k=1,
                                include_values=True)
        
        return response
        
def cossim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return abs(cos_sim) 