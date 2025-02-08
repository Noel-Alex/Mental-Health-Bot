import os
import asyncio
from langchain_groq import ChatGroq
from langchain.agents import Tool, initialize_agent, AgentType
from dotenv import load_dotenv
import utils.query as query

load_dotenv()


# --------------------------
# Asynchronous helper functions
# --------------------------

async def provide_self_care_tip(query_str: str) -> str:
    """
    Provides a generic self-care tip.
    Input: A description of the problem faced by the user.
    """
    prompt_template = (
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
    response = await asyncio.to_thread(query.query, prompt_template + query_str)
    return response


async def store_data(structured_data: str) -> None:
    """
    Stores data about the user's mood or stress along with a timestamp.
    """
    await asyncio.to_thread(query.generate_embeddings, structured_data)


async def get_emergency_resources(user_input: str) -> str:
    """
    Provides emergency resource information.
    """
    resources = (
        "If you're in immediate danger or experiencing a crisis, please call your local emergency services immediately. "
        "In the United States, dial 911. If you need mental health crisis support, consider contacting the National Suicide Prevention Lifeline at 988 or visit https://988lifeline.org/. "
        "If you're located elsewhere, please consult local emergency resources."
    )
    return resources


async def reflective_listening(user_input: str) -> str:
    """
    Reflects the user's feelings back to them.
    For demonstration, this function simply echoes a supportive message.
    """
    prompt_template = f"""
        You are an empathetic mental health assistant. Respond to the user by:  
    1. Explicitly naming their emotion ("I hear you’re feeling [emotion]").  
    2. Validating it ("That’s completely normal/valid").  
    3. Offering support ("I’m here to help").  
    4. Suggesting a small action ("Would you like to try [activity]?").  
    5. Ending with an open question ("How can I support you right now?").  

    The user’s current emotion is: {user_input}.  

    Example response for "frustrated":  
    \"I sense you’re feeling frustrated, and that’s totally understandable. It’s okay to feel this way when things don’t go as planned. Would it help to take a short walk or brainstorm solutions together?\"  """



async def summarize_conversation(conversation_history: str) -> str:
    """
    Summarizes the conversation history.
    """
    summary = (
        "From our conversation, it appears that you're experiencing some distress. "
        "Remember, while I'm here to support you, talking to a mental health professional can provide more tailored help."
    )
    return summary


async def user_info_retrieval(query_str: str) -> str:
    """
    Retrieves important info about the user.
    """
    result = await asyncio.to_thread(query.query, query_str)
    return result


# --------------------------
# Main asynchronous function
# --------------------------

async def main():
    # Initialize the language model
    llm = ChatGroq(groq_api_key=os.getenv("GROQ"), model_name="llama-3.3-70b-specdec")

    # Wrap the functions as Tools for the agent
    tools = [
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
                "Goes through all important info about the user and answers the query. "
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
        max_iterations=7
    )

    # Start the asynchronous conversation loop.
    print("Mental Health Chatbot (Prototype)")
    print("Type 'quit' to exit.\n")
    conversation_history = ""
    loop = asyncio.get_running_loop()

    while True:
        # Run blocking input in a thread to avoid blocking the event loop.
        user_input = await loop.run_in_executor(None, input, "You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Thank you for the conversation. Take care!")
            break

        conversation_history += f"User: {user_input}\n"

        # If your agent does not support async, run it in a thread.
        response = await loop.run_in_executor(None, agent.run, user_input)

        conversation_history += f"Chatbot: {response}\n"
        print("Chatbot:", response)
        print("\n---\n")


if __name__ == "__main__":
    # Run the asynchronous main function.
    asyncio.run(main())
