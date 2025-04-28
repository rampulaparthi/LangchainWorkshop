from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Prompt 1: Summarization
joke_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Prompt 2: sentiment
summarize_prompt = ChatPromptTemplate.from_template(
    "what are these jokes about : {input}?"
)

# Prompt 3: Translation
translate_prompt = ChatPromptTemplate.from_template(
    "Translate this to French: {input}"
)

# Step 1: Summarize input
joke_chain = joke_template | model

wrap_as_input = RunnableLambda(lambda x: {"input": x})


# Step 2: Translate output from summarization
summarize_chain = summarize_prompt | model

# Step 3: Translate output from summarization
translate_chain = translate_prompt | model

# Step 4: Combine the chains
# Combine: Chain summarization -> translation
chained_pipeline = joke_chain | wrap_as_input | summarize_chain | wrap_as_input | translate_chain

# Run
result = chained_pipeline.invoke({"topic": "lawyers", "joke_count": 3})
print(result.content)