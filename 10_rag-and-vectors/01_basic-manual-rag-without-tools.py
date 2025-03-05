import os
import glob
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

openai = OpenAI()

context = {}

employees = glob.glob("./knowledge-base/employees/*")
for employee in employees:
    name = employee.split(" ")[-1][:-3]
    doc = ""
    with open(employee, "r", encoding="utf-8") as file:
        context[name] = file.read()

products = glob.glob("./knowledge-base/products/*")
for product in products:
    name = product.split(os.sep)[-1][:-3]
    doc = ""
    with open(product, "r", encoding="utf-8") as file:
        context[name] = file.read()

print(context.keys())

system_message = "You are an export in answering accurate questions about Insurellm, the Insurance Tech company. Give brief, accurate answers. If you don't know the answer, say 'I don't know' politely. Do not make things up if you haven't been provided with relevant context."

def get_relevant_context(message):
    relevant_context = []
    for context_title, context_details in context.items():
        if context_title.lower() in message.lower():
            relevant_context.append(context_details)
    return relevant_context

def add_context(message):
    relevant_context = get_relevant_context(message)
    if relevant_context:
        message += "\n\nThe following additional context might be relevant in answering thie question:\n\n".join(relevant_context)
        for relevant in relevant_context:
            message += f"{relevant}\n\n"
    return message

def ask_question(message):
    message = add_context(message)
    messages = [
        {"role": "system", "content": system_message}, 
        {"role": "user", "content": message}]
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    return response.choices[0].message.content

# print(ask_question("What is the role of the CEO?"))
# print()
# print(ask_question("Who is Alex Lancaster??")) 
# print()
# print(ask_question("What is Homellm?"))
# print()

# print(ask_question("How much Blake is earning?"))
# print()
# print(ask_question("What is the salary of Blake"))
# print()
# print(ask_question("What is the base salary of Blake")) 
# print()
# print(ask_question("What is latest compensation of of Blake")) 
# print()
# print(ask_question("What is compensation history of of Blake"))

print()
print(ask_question("What is the salary of Blak")) # Unable to answer, due to spelling mistake

print()
print(ask_question("Who is Alex??")) # Unable to answer, because we have index on last name

