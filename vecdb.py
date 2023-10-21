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
        
def cossimhist(vec1, vec_dict:dict, thresh=0.5):
    """
    A function to calculate cosine similarity between a vector and a list of vectors.
    Designed to take a vector and a list of vectors. 
    Returns the n most similar vectors to the input one + answers to them. 
    """
    n_prompts_answers = {}

    for key in vec_dict.keys(): 
        cos_sim = np.dot(vec1, vec_dict[key][0])/(np.linalg.norm(vec1)*np.linalg.norm(vec_dict[key][0]))
        if cos_sim >= thresh:
            n_prompts_answers[key] = vec_dict[key][1]

    return n_prompts_answers

#a function to retreive all the previous prompts from some history 
def retreive_hist(inp:dict):
    """
    A function to retreive the entire chat history from an input dictionary.
    Specific to the pipeline of the project.
    """
    vec_dict = {}
    history = inp['history'] #a list of all prompt-answer pairs

    for pair in history:
        vec_dict[pair['prompt']] = [pair['vectorized_prompt'], pair['answer']]

    return vec_dict