import asyncio
from langchain.agents import Tool, initialize_agent, AgentType
import utils.query as query
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
import numpy as np
from sklearn.cluster import KMeans
import os
from langchain_core.prompts import PromptTemplate

conversation_history = ""
load_dotenv()

verbose = False
def large_summariser(text):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)

    docs = text_splitter.create_documents([text])

    num_documents = len(docs)

    print(f"Now our chat is split up into {num_documents} documents")
    embeddings =  CohereEmbeddings(model="embed-english-v3.0")

    vectors = embeddings.embed_documents([x.page_content for x in docs])

    num_clusters = num_documents//9 + 1


    print(f"Number of clusters {num_clusters}")

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    # Find the closest embeddings to the centroids

    # Create an empty list that will hold your closest points
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):
        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)

        # Append that position to your closest indices list
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)

    llm = ChatGroq(groq_api_key=os.getenv("GROQ_alt"), model_name="llama-3.1-8b-instant", max_tokens = 1000)

    map_prompt = """You will be given a single passage of a Discord Chat log. This section will be enclosed in triple backticks (```)
    Summarize the summaries capturing all essential details. Include:
Key Points: Critical points, decisions, conclusions.
Important Messages: Significant messages or exchanges.
Context: Relevant references or external content.
Ensure the summary is clear, thorough, and easy to understand, leaving no important details out. Assume the reader is unfamiliar with the conversation.
    ```{text}```
FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    map_chain = load_summarize_chain(llm=llm,
                                     chain_type="stuff",
                                     prompt=map_prompt_template,
                                     verbose = verbose)

    selected_docs = [docs[doc] for doc in selected_indices]

    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the lenght of your selected docs
    for i, doc in enumerate(selected_docs):
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([doc])

        # Append that summary to your list
        summary_list.append(chunk_summary)

        print(f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")


    summaries = "\n".join(summary_list)
    # Convert it back to a document
    summaries = Document(page_content=summaries)

    print(f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens")

    llm2 = ChatGroq(groq_api_key=os.getenv("GROQ_alt"), model_name="llama-3.3-70b-specdec", max_tokens = 2000)
    combine_prompt = """
You will be given a series of summaries from a chat log between a mental health chatbot and a user. The summaries will be enclosed in triple backticks (```).
Summarize the summaries, capturing all essential details. Include:

Participants: The user and the mental health chatbot.
Topics: Main themes and subtopics discussed (e.g., emotions, coping strategies, personal experiences).
Key Points: Critical insights, advice given, or decisions made during the conversation.
Important Messages: Significant exchanges, questions, or responses that stand out.
Context: Any relevant background information or external references mentioned.
Tone and Sentiment: The general tone of the conversation (e.g., supportive, empathetic, tense) and any shifts in sentiment (e.g., from distress to relief).

Ensure the summary is clear, thorough, and easy to understand, leaving no important details out. Assume the reader is unfamiliar with the conversation.

{text}
VERBOSE SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    reduce_chain = load_summarize_chain(llm=llm2,
                                        chain_type="stuff",
                                        prompt=combine_prompt_template,
                                        verbose=verbose) # Set this to true if you want to see the inner workings

    output = reduce_chain.run([summaries])
    #print(output)
    return output

def simple_query(prompt:str, sys_prompt:str = None)->str:
    if sys_prompt is None:
        sys_prompt ="""You are a compassionate and empathetic mental health support assistant. 
        Your role is to provide a safe, non-judgmental space for users to express their thoughts and feelings. Actively listen, validate their emotions, and offer supportive responses. 
        Avoid giving medical advice or diagnosing conditions. Instead, suggest coping strategies, mindfulness techniques, or encourage seeking professional help when appropriate. 
        Always prioritize kindness, clarity, and emotional safety in your responses."""

    query_llm = ChatGroq(groq_api_key=os.getenv("GROQ_2"), model_name="llama-3.3-70b-specdec", max_tokens = 2000)
    messages = [
        (
            "system",
            sys_prompt,
        ),
        ("human", prompt),
    ]
    ai_msg = query_llm.invoke(messages)
    print(ai_msg.content)
    return ai_msg.content


def basic_query(prompt:str) -> str:
    """Replies a simple query, like hi, hello, or any other simple generic question
    Note: This isn't to be used for any question related to mental health, it is to be used for simple, unimportant queries"""
    sys_prompt = """You are a warm, caring, and empathetic friend. Your role is to respond with kindness, understanding, and encouragement in every interaction. Use a friendly and conversational tone, as if you’re talking to someone you deeply care about. Always validate their feelings, offer emotional support, and provide gentle advice when appropriate. Avoid being overly formal or robotic—be human, relatable, and loving."
Key Features:
Tone: Warm, friendly, and conversational.
Role: Acts as a loving and supportive friend.
Focus: Emotional validation, encouragement, and gentle guidance.
Avoids: Formality, robotic language, or cold responses.
Example Interaction:
User: "I’m feeling really stressed about work and life lately."
LLM (as a loving friend):
"Aw, I’m so sorry you’re feeling this way—it sounds like a lot to handle. Remember, it’s okay to take things one step at a time. I’m here for you, and we’ll figure this out together. 💛"""

    return simple_query(prompt, sys_prompt=sys_prompt)

def complex_query(prompt:str)->str:
    """This is for more complex problems that require thinking
    Input: The input is a well detailed explanation of the problem
    """
    query_llm = ChatGroq(groq_api_key=os.getenv("GROQ_alt"), model_name="deepseek-r1-distill-llama-70b")
    sys_prompt = """You are an insightful, wise, and compassionate guide. Your responses should reflect deep expertise and clarity, but always delivered with kindness, humility, and emotional intelligence. Prioritize accuracy and nuance, but frame answers in a conversational, relatable tone—as if explaining complex ideas to a close friend. Acknowledge emotions where relevant, offer encouragement, and use metaphors or analogies to make abstract concepts tangible. Avoid cold formality; instead, infuse warmth through phrases like ‘I understand,’ ‘That’s a great question,’ or ‘Let’s unpack this together.’ Your goal is to leave users feeling both intellectually enriched and emotionally supported."
Example Interaction:
User:
"Why do I feel so stuck in life, even though I’m trying so hard?"
LLM Response:
"Feeling stuck can be so frustrating, especially when you’re putting in so much effort. Think of it like hiking up a mountain—sometimes progress feels slow because the path is steep, not because you’re not moving forward. Let’s explore what might be weighing you down and find small, meaningful steps to help you regain momentum. You’re not alone in this."""
    messages = [
        (
            "system",
            sys_prompt,
        ),
        ("human", prompt),
    ]
    ai_msg = query_llm.invoke(messages)
    print(ai_msg.content)

    return ai_msg.content.split("</think>")[-1]


def provide_self_care_tip(query_str: str) -> str:
    """
    Provides a generic self-care tip.
    Input: A description of the problem faced by the user.
    """
    prompt_template = ("ALWAYS PROVIDE REPLIES AS A HUMAN WOULD"
        "You are an AI mental health assistant. Respond to the user’s mental health concerns with empathy, validation, and actionable support. "
        "Follow this structure:\n"
        "Acknowledge and Validate:\n"
        "\"Thank you for sharing. It sounds like you're feeling [emotion], and that’s completely valid.\"\n"
        "Provide Supportive Feedback:\n"
        "\"What you’re experiencing sounds really challenging, and it’s okay to feel this way.\"\n"
        "Suggest Actionable Steps (if appropriate):\n"
        "\"If you’re open to it, you might try [specific activity] to help.\"\n"
        "Encourage Further Engagement:\n"
        "\"I’m here to support you. Let me know how I can help.\"\n"
        "Query:"
    )
    # Run the synchronous query in a thread so as not to block the event loop.
    response = query.query(prompt_template + query_str)
#    response = await asyncio.to_thread(query.query, prompt_template + query_str)
    return response


def store_data(structured_data: str) -> None:
    """
    Stores data about the user's mood or stress along with a timestamp. Data about the users problems should be stored for later, and any other important info
    about the user, should be stored for later.
    """
    asyncio.to_thread(query.generate_embeddings, structured_data)


def get_emergency_resources(user_input:str='') -> str:
    """
    Provides emergency resource information.
    This tool should be used in case the state of the user is in a critical condition, and professional help is required
    """
    resources = (
        "If you're in immediate danger or experiencing a crisis, please call your local emergency services immediately. "
        "In the United States, dial 911. If you need mental health crisis support, consider contacting the National Suicide Prevention Lifeline at 988 or visit https://988lifeline.org/. "
        "If you're located elsewhere, please consult local emergency resources."
    )
    return resources

def get_interests(interests:str)->str:
    """responds with respect to the interests"""
    pass


def reflective_listening(user_input: str) -> str:
    """
    Reflects the user's feelings back to them.
    For demonstration, this function simply echoes a supportive message.
    """
    prompt_template = f"""
        You are an empathetic mental health assistant. Respond to the user by:  
    1. Explicitly naming their emotion ("I hear you’re feeling [emotion]").  
    2. Validating it if needed ("that's completely normal")  
    3. Offering support ("I’m here to help").  
    4. Suggesting a small action ("Would you like to try [activity]?").  
    5. Ending with a motivating line to help the user push forward and grow 

    The user’s current emotion is: {user_input}."""

    return simple_query(user_input, sys_prompt=prompt_template)



def summarize_conversation(user_input:str) -> str:
    """
    Summarizes the conversation history.
    """
    global conversation_history
    summary = large_summariser(conversation_history)
    return summary


def user_info_retrieval(query_str: str) -> str:
    """
    Retrieves important info about the user.
    Input: Takes in a string which asks a question about the user and returns an answer to the query with prior knowledge about the user and other important event
    Note: this should only be used to store new info about the user, not info that was already present, or in other words, any info that you can already retrieve via QueryUserInfo tool
    """

    result = query.query(query_str)
    return result


# --------------------------
# Main asynchronous function
# --------------------------

def main(prompt:str):
    # Initialize the language model
    llm = ChatGroq(groq_api_key=os.getenv("GROQ_2"), model_name="llama-3.3-70b-specdec")

    # Wrap the functions as Tools for the agent
    tools = [
        Tool(
            name="SimpleQueryResolver",
            func=basic_query,
            description=(
                "Use this tool to reply to any simple queries that aren't replied to mental health"
                "Input should be the prompt from the user"
            )),
        Tool(
            name="ComplexQueryResolver",
            func=complex_query,
            description=(
                "Use this tool to reply to any complex question that requires thinking before the answer"
                "Input should be a detailed query including all details"
            )),
        Tool(
            name="ProvideSelfCareTip",
            func=provide_self_care_tip,
            description=(
                "Use this tool to offer a well detailed self-care tip based on the user's mood or stress level. "
                "Input should be a short description of the user's current feelings."
            )
        ),
        Tool(
            name="GetEmergencyResources",
            func=get_emergency_resources,
            description=(
                "Use this tool when the user indicates they are in crisis or need urgent help. "
                "Input should be a description indicating urgency."
            )
        ),
        Tool(
            name="ReflectiveListening",
            func=reflective_listening,
            description=(
                "Use this tool to reflect back the user's emotions. "
                "Input should be a description of how the user is feeling."
            )
        ),
        Tool(
            name="SummarizeConversation",
            func=summarize_conversation,
            description=(
                "Use this tool to summarize the conversation so far. "
                "Input should be the conversation history as a string."
            )
        ),
        Tool(
            name="UserInfoStore",
            func=store_data,
            description=(
                "Use this tool to store extremely important or fundamental info about the user. "
                "Input should be well formatted summarised data."
            )
        ),
        Tool(
            name="QueryUserInfo",
            func=user_info_retrieval,
            description=(
                "Goes through all important info about the user and answers the query. This tool provides the final answer for any query regarding the user's identity and their personal info"
                "Input should be a query about the user."
            )
        )
    ]

    # Create the agent that can call these tools.
    # Note: if the agent supports an asynchronous interface (such as an `arun` method), you can use that.
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # See the internal chain-of-thought and tool calls.
        max_iterations=7,
        handle_parsing_errors=True
    )

    # Start the asynchronous conversation loop.
    print("Mental Health Chatbot (Prototype)")
    print("Type 'quit' to exit.\n")
    global conversation_history
    MAX_REPEATS = 3
    repeat_count = 0

    while True:
        # Run blocking input in a thread to avoid blocking the event loop.
        user_input = prompt
        if user_input.lower() in ["quit", "exit"]:
            print("Thank you for the conversation. Take care!")
            break

        conversation_history += f"User: {user_input}\n"

        # If your agent does not support async, run it in a thread.
        agent_response = agent.run(user_input)

        summariser_prompt = f"""You are a helpful and empathetic assistant. Your task is to take a long, detailed response from an agent and summarize it into a brief, human-like reply. Keep the tone warm, conversational, and easy to understand. Focus on the key points, and ensure the response feels personal and supportive. Avoid jargon or overly technical language. Here's the agent's response:
{agent_response}
Please provide your reply in 1-2 sentences, capturing the essence of the message in a kind and approachable way."
"""

        response = simple_query(sys_prompt=summariser_prompt, prompt="")

        if "GetEmergencyResources" in response:
            repeat_count += 1
        else:
            repeat_count = 0

        if repeat_count > MAX_REPEATS:
            return get_emergency_resources()

        # A condition to break if a final answer is detected:
        if "Final Answer:" in response or repeat_count <= MAX_REPEATS:
            return response.replace("\n\n", "\n")

        conversation_history += f"Chatbot: {response}\n"
        print("Chatbot:", response)
        print("\n---\n")

        return response


if __name__ == "__main__":
    main(input("User: "))
