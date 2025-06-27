from agents import Agent, OpenAIChatCompletionsModel, Runner
from openai import AsyncOpenAI
from agents.run import RunConfig
from dotenv import load_dotenv
import os
import chainlit as cl
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client= AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model= OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client,
    
)

config= RunConfig(
    model=model,
    tracing_disabled=True,

)

agent = Agent(
    name='Assistant',
    instructions='you are a helpful assistant.',
)

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content
    result = await Runner.run(agent, user_input, run_config=config)
    await cl.Message(content=result.final_output).send()