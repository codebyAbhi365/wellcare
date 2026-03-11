import asyncio
import cognee
import os

os.environ["LLM_MODEL"] = "llama3:latest"
os.environ["EMBEDDING_MODEL"] = "qwen3-embedding:8b"

async def main():

    data = [
        "Diabetes is a chronic disease that occurs when blood glucose is too high.",
        "Insulin is a hormone made by the pancreas.",
    ]

    print("Adding knowledge to Cognee...")
    await cognee.add(data)

    print("Building knowledge graph...")
    await cognee.cognify()

    print("\nAsking question...\n")

    result = await cognee.search("What is diabetes?")

    print("Response:")
    print(result)


asyncio.run(main())