import os
from dotenv import load_dotenv # Added for loading API key safely

# Removed redundant operator import
# from operator import itemgetter # Not used directly here

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# Import the Google Generative AI chat model class
from langchain_google_genai import ChatGoogleGenerativeAI
# Using ChatMessageHistory from langchain-community for simple in-memory storage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory # Base class for type hinting
# Removed unused ConversationBufferMemory import

# --- Configuration ---


# Get the Google API Key from environment variables
google_api_key = "AIzaSyDE_7SpoXiWdsPU3nOvKAFlpdWxSYGiOP4"

# --- !! Security Warning !! ---
# Avoid hardcoding keys directly in your code like this:
# google_api_key = 'YOUR_GOOGLE_API_KEY_HERE' # <--- Avoid this in production/shared code

if not google_api_key:
    print("Error: GOOGLE_API_KEY not found.")
    print("Please ensure you have a .env file with GOOGLE_API_KEY=your_key or set the environment variable.")
    exit()

# --- Model Initialization ---
# Initialize the ChatGoogleGenerativeAI model
# Specify the model name, e.g., "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"
try:
    # Pass the key explicitly if needed, though it often picks it up from the environment
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Using Flash for speed/cost efficiency
        google_api_key=google_api_key
        # Optional: Add safety settings or other configurations if required
        # safety_settings={...}
        # convert_system_message_to_human=True # Sometimes helpful
    )
except Exception as e:
    print(f"Error initializing Google Generative AI model: {e}")
    exit()

# --- Prompt Template ---
# Define the structure of the prompt sent to the model. (Remains the same)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the user's questions clearly and concisely."),
        MessagesPlaceholder(variable_name="chat_history"), # Where the history object's messages will be inserted
        ("human", "{input}"),
    ]
)

# --- Memory Store ---
# Use a dictionary to store ChatMessageHistory objects per session_id (Remains the same)
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
# RunnableWithMessageHistory works seamlessly with the new model (Remains the same structure)
conversational_chain = RunnableWithMessageHistory(
    prompt | model,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history", # This name matches the MessagesPlaceholder
)

# --- Interaction Loop ---
print("Simple Gemini Chatbot Ready! Type 'quit' to exit.") # Updated message
# Use a unique session ID for this run
session_id = "gemini_chat_123" # Example session ID (updated for clarity)

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Bot: Goodbye!")
        break

    # Prepare the input for the chain
    chain_input = {"input": user_input}

    # Define the configuration for the run, including the session_id
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