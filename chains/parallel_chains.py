from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}.")
    ]
)


# Define pros analysis step
def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "Given these features: {features}, list top 3 pros of these features.")
        ]
    )
    return pros_template.format_prompt(features=features)


# Define cons analysis step
def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human","Given these features: {features}, list top 3 cons of these features.")
        ]
    )
    return cons_template.format_prompt(features=features)


# Simplify branches with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Combine pros and cons into a final review
def combine_branches(dict):
    pros=dict["branches"]["pros"]
    cons=dict["branches"]["cons"]
    return f"Pros:\n{pros}\n\nCons:\n{cons}"

# Use the named function in RunnableLambda
RunnableLambda(combine_branches)

# Create the combined chain using LCEL
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain}) #runs both of these branches in parallel
    | RunnableLambda(lambda x: combine_branches(x))
)

# Run the chain
result = chain.invoke({"product_name": "Iphone 14 Pro Max"})

# Output
print(result)
