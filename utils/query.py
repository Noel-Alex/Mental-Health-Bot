import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import Document
from langchain import hub
from llama_index.core import PromptTemplate

from datetime import datetime



#langchain_prompt = hub.pull('rlm/rag-prompt')
#Load Tokens
load_dotenv()
GROQ = os.getenv('GROQ')
HF_TOKEN = os.getenv('HF_TOKEN')
cohere_api_key = os.getenv('COHERE_API_KEY')


llm_model = "llama-3.1-8b-instant"


def create_folders_and_file(folder_path, filename) ->str:
  """
  Creates folders and subfolders if they don't exist and writes content to a file in the deepest folder.

  Args:
      folder_path (str): Path to the top-level folder.
      filename (str): Name of the file to create in the deepest folder.
      content (str, optional): Content to write to the file. Defaults to "This is some text".
  """

  # Ensure path is a string
  if not isinstance(folder_path, str):
    raise TypeError("folder_path must be a string")

  # Create folders using os.makedirs with exist_ok=True to handle existing directories
  try:
    os.makedirs(folder_path, exist_ok=True)
  except OSError as e:
    print(f"Error creating directories: {e}")
    return

  # Create the file with full path
  full_path = os.path.join(folder_path, filename)
  try:
    with open(full_path, 'w') as f:
        pass
    print(f"Successfully created file: {full_path}")
    return full_path
  except OSError as e:
    print(f"Error creating file: {e}")

def generate_embeddings(data, embedding_path:str="./embeddings", purpose="quickstart")->None:
    print("Generating embeddings...")

    load_dotenv()
    # Initialize embeddings
    embeddings = CohereEmbedding(
        api_key=cohere_api_key,
        model_name="embed-english-light-v3.0",
        input_type="search_query",

    )

    Settings.embed_model = embeddings

    #    Settings.embed_model = HuggingFaceEmbedding(
    #    model_name = 'nomic-ai/nomic-embed-text-v1'
    #    )

#    documents = SimpleDirectoryReader(documents_path).load_data()
    document = Document(text = data)
    document.metadata={"date":datetime.utcnow()}
    db = chromadb.PersistentClient(path=embedding_path)
    # create collection
    chroma_collection = db.get_or_create_collection(purpose)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create your index
    index = VectorStoreIndex.from_documents(
        [document], storage_context=storage_context
    )

    print('Done generating embeddings')



def query(prompt:str, embedding_path:str="./embeddings", purpose="quickstart") -> str:
    model = "llama-3.3-70b-specdec"
    llm = Groq(model=model, api_key=GROQ)
    Settings.llm = llm

    # Initialize embeddings
    embeddings = CohereEmbedding(
        api_key=cohere_api_key,
        model_name="embed-english-light-v3.0",
        input_type="search_query",

    )
    Settings.embed_model = embeddings

    # initialize client
    db = chromadb.PersistentClient(path=embedding_path)

    # get collection
    chroma_collection = db.get_or_create_collection(purpose)

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    qa_prompt_tmpl = (
        "The following is data on the patient you are treating, THE FOLLOWING TEXT IS NOT ABOUT YOU, IT'S ABOUT THE PERSON YOU ARE TALKING TO, WHO YOU ARE HELPING IMPROVE THEIR MENTAL HEALTH\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        """Given the context and the fact that you are an expert Psychologist treating this patient, and you are talking to a patient and analyzing any verbal queues, and making well organised and structured notes
    on your patients well being and emotions, be as human as possible in your responses, be kind and gentle, do not talk like a robot.hey, how are you?"""
        "answer the query.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt = PromptTemplate(qa_prompt_tmpl)

    # load your index from stored vectors
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    query_engine = index.as_query_engine(summary_template = qa_prompt)

    response = query_engine.query(prompt)
    return response