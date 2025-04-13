import os
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory # Used for the basic memory store

# --- Configuration ---
# Load environment variables (especially GROQ_API_KEY)

# Check if the API key is loaded (optional but good practice)
import os

from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
# Import the simpler history class
# Use InMemoryChatMessageHistory for this example, part of langchain-community
# If you haven't installed it: pip install langchain-community
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory # Base class for type hinting



# Check if the API key is loaded (optional but good practice)
groq_api_key = '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'
if not groq_api_key:
    print("Error: GROQ_API_KEY not found in environment variables.")
    exit()

# --- Model Initialization ---
# Use a specific Groq model, e.g., llama3-8b-8192 or mixtral-8x7b-32768
model = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_api_key)

# --- Prompt Template ---
# Define the structure of the prompt sent to the model.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the user's questions concisely."),
        MessagesPlaceholder(variable_name="chat_history"), # Where the history object's messages will be inserted
        ("human", "{input}"),
    ]
)

# --- Memory Store ---
# Use a dictionary to store ChatMessageHistory objects per session_id
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves or creates a ChatMessageHistory object for a given session ID.
    """
    if session_id not in store:
        # Create a new ChatMessageHistory instance for the session
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Create the Chain with Message History ---
# RunnableWithMessageHistory works well with objects that implement BaseChatMessageHistory
# (like the ChatMessageHistory we are returning from get_session_history)
conversational_chain = RunnableWithMessageHistory(
    prompt | model,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history", # This name matches the MessagesPlaceholder
)

# --- Interaction Loop ---
print("Simple Groq Chatbot Ready! Type 'quit' to exit.")
# Use a unique session ID for this run
session_id = "user123" # Example session ID

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Bot: Goodbye!")
        break

    # Prepare the input for the chain
    chain_input = {"input": user_input}

    # Define the configuration for the run, including the session_id
    # This tells RunnableWithMessageHistory which history object to use from the 'store'
    config = {"configurable": {"session_id": session_id}}

    try:
        # Invoke the chain
        response = conversational_chain.invoke(chain_input, config=config)

        # Extract the content from the AIMessage response
        print(f"Bot: {response.content}")

    except Exception as e:
        print(f"An error occurred: {e}")
        # You might want to add more robust error handling or logging here
        # break # Uncomment if you want to stop on error

# Optional: Inspect the final memory state for the session
# print("\n--- Final Memory State ---")
# final_history = get_session_history(session_id)
# print(final_history.messages)