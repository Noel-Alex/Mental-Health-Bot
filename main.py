'''import utils.query as query
from crewai_tools import LlamaIndexTool
import os'''

import os
from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from llama_index.llms.groq import Groq


load_dotenv()

llm = Groq(model="llama-3.3-70b-specdec", api_key=os.getenv("GROQ"))
function_calling_llm = Groq(model="	llama-3.3-70b-versatile", api_key=os.getenv("GROQ_alt"))
thinking_model =Groq(model="deepseek-r1-distill-llama-70b-specdec", api_key=os.getenv("GROQ_alt"))


'''query_engine = query.query()


query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="infy Query Tool",
    description="This tool is used to ",
)'''


therapist = Agent(
    role="Expert Mental Health Therapist",
    goal="To help the user to feel better about herself, solve any issues that are troubling her mind, and ensure the long term mental wellbeing of your patient",
    backstory="""You are the world's greatest therapist, you have been given the task to analyse the report on the patient's mental health and mood given to you
    by your assistant, and decide on the best possible decision to suggest or thing to say to help the patient feel better""",
    verbose=True,
    allow_delegation=True,
#    tools=[query_tool],
    llm=llm,
    function_calling_llm=thinking_model,
    memory = True
)

analyser = Agent(
    role="Expert Psychological Analyser",
    goal="To accurately assess any emotional queues in the chat with the user and generate structured notes on the patient",
    backstory="""You are an expert Psychologist, and you are talking to a patient and analyzing any verbal queues, and making well organised and structured notes
    on your patients well being and emotions""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    function_calling_llm = function_calling_llm,
    memory = True
)

task1 = Task(
    description="""To analyse the given prompt from the user and analyse the mood of the user, understand how the day of the user went, figure out all the issues the user is facing mentally
    and absolutely anything else of note that could be used in a psychological analysis and anything that could be needed to understand the user to help their day to day mental health improve.
    Make all these notes in a structured format.
    USER:
    Hey I've been having a horrible day lately, work has been hectic, the weather is horrible and I have such a horrible cold right now, can you please help me out?""",

    expected_output="Full analysis report in bullet points",
    agent=analyser,
)

task2 = Task(
    description="""Using the insights provided, Talk to the patient and analyze what to say to the user or suggest to help the patient feel better keeping in mind
    both the long term and short term well being of the user""",
    expected_output="A satisfactory reply to the user that is very humane, kind, gentle and helps the user feel better about themselves and help them both short and long term with respect to mental health",
    agent=therapist,
)

crew = Crew(
    agents=[therapist, analyser],
    tasks=[task1, task2],
)

result = crew.kickoff()

print("######################")
print(result)