

from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
import uuid
import boto3


load_dotenv(override=True)


# Initialize DynamoDB client using environment variables from the .env file
dynamodb = boto3.client('dynamodb')


history = DynamoDBChatMessageHistory(
    table_name="SessionTable",
    session_id=str(uuid.uuid4()),
)

model = ChatOpenAI(model="gpt-4o")

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    history.add_user_message(human_input)

    ai_response = model.invoke(history.messages)
    history.add_ai_message(str(ai_response.content))

    print(f"AI: {ai_response.content}")
