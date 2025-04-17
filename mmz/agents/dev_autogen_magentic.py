# Enable human-in-the-loop mode
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import UserMessage
from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings
from semantic_kernel.memory.null_memory import NullMemory
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_agentchat.ui import Console
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings


host = "http://127.0.0.1:11434/v1"
#model_name = 'qwen2.5-coder:7b-instruct-q4_K_M'
#model_name = 'starling-lm:latest'
#model_name = 'gemma3:12b'
#model_name = 'llama3.2:latest'
model_name = 'granite3.2-vision'


async def example_usage_hil():
    #client = OpenAIChatCompletionClient(model="gpt-4o")
    #sk_client = OllamaChatCompletion(
    # TODO: This was confusing - note this somewhere: don't us llama python api, but openAI and 

    client = OpenAIChatCompletionClient(
        model=model_name,
        base_url=host,
        api_key='placeholder',
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        }
        #host=host, #"http://localhost:11434",
        #ai_model_id=model_name #"llama3.2:latest",
    )
    #ollama_settings = OllamaChatPromptExecutionSettings(
    #    options={"temperature": 0.5},
    #)
    #model_client = SKChatCompletionAdapter(
    #    sk_client,
    #    #kernel=Kernel(memory=NullMemory()),
    #    prompt_settings=ollama_settings
    #)
    
    # to enable human-in-the-loop mode, set hil_mode=True
    m1 = MagenticOne(client=client, hil_mode=True)
    #m1 = MagenticOne(client=model_client, hil_mode=True)
    #task = "Write a Python script to fetch data from an API."
    task = "Write a Python script to list the running processes on a linux system"
    result = await Console(m1.run_stream(task=task))
    print(result)


if __name__ == "__main__":
    asyncio.run(example_usage_hil())

