import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import UserMessage
from autogen_ext.models.semantic_kernel import SKChatCompletionAdapter
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaChatPromptExecutionSettings
from semantic_kernel.memory.null_memory import NullMemory


host = "http://127.0.0.1:11434"
model_name = 'qwen2.5-coder:7b-instruct-q4_K_M'

async def main() -> None:
    sk_client = OllamaChatCompletion(
        host=host, #"http://localhost:11434",
        ai_model_id=model_name #"llama3.2:latest",
    )
    ollama_settings = OllamaChatPromptExecutionSettings(
        options={"temperature": 0.5},
    )

    model_client = SKChatCompletionAdapter(
        sk_client, kernel=Kernel(memory=NullMemory()), prompt_settings=ollama_settings
    )

    # Call the model directly.
    model_result = await model_client.create(
        messages=[UserMessage(content="What is the capital of France?", source="User")]
    )
    print(model_result)

    # Create an assistant agent with the model client.
    assistant = AssistantAgent("assistant", model_client=model_client)
    # Call the assistant with a task.
    result = await assistant.run(task="What is the capital of France?")
    print(result)


asyncio.run(main())
