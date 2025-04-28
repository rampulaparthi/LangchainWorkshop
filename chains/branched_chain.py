from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Generate a response addressing this positive feedback: {feedback} as a paragraph and do not write it in a form of a letter."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Generate a response addressing this negative feedback: {feedback}. Please include an apology and a contact method for further assistance."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Escalate this feedback to the support team: {feedback}.",
        ),
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),
    ]
)

positive_feedback_chain = positive_feedback_template | model | StrOutputParser()
negative_feedback_chain = negative_feedback_template | model | StrOutputParser()
neutral_feedback_chain = neutral_feedback_template | model | StrOutputParser()
escalate_feedback_chain = escalate_feedback_template | model | StrOutputParser()


# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x, positive_feedback_chain
    ),
    (
        lambda x: "negative" in x, negative_feedback_chain
    ),
    (
        lambda x: "neutral" in x, neutral_feedback_chain
    ),
    escalate_feedback_chain
)

# Create the classification chain
classification_chain = classification_template | model | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches

# Run the chain with an example review
# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = "The product is okay. It works as expected but nothing exceptional."
result = chain.invoke({"feedback": review})

# Output the result
print(result)
