from llama_index.llms.groq import Groq
import os
from llama_index.core import Settings
import utils.query as query
from dotenv import load_dotenv

load_dotenv()



while True:
    prompt = input("Enter prompt")
    if prompt =="q":
        break
    print(query.query(prompt))

