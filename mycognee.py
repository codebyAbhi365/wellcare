import os
import asyncio

# LLM Configuration
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["LLM_MODEL"] = "gemini/gemini-1.5-flash"
os.environ["LLM_API_KEY"] = "AIzaSyB0EbSEdDcfu3bgUzILq3p_mk4isMxpnQI"
os.environ["LLM_API_VERSION"] = "v1"
os.environ["LLM_ENDPOINT"] = ""

# Embedding Configuration
os.environ["EMBEDDING_PROVIDER"] = "gemini"
os.environ["EMBEDDING_MODEL"] = "gemini/gemini-embedding-001"
os.environ["EMBEDDING_API_KEY"] = "AIzaSyB0EbSEdDcfu3bgUzILq3p_mk4isMxpnQI"
os.environ["EMBEDDING_API_VERSION"] = "v1"
os.environ["EMBEDDING_ENDPOINT"] = ""
os.environ["EMBEDDING_DIMENSIONS"] = "3072"

# Tokenizer Requirement
os.environ["HUGGINGFACE_TOKENIZER"] = "google/gemma-2b"

# Import cognee AFTER setting environment variables
import cognee

async def main():
    # Clean system to ensure fresh schema with new dimensions
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    text = "Cognee turns documents into AI memory."
    await cognee.add(text)

    await cognee.cognify()

    results = await cognee.search(
        query_text="What does Cognee do?"
    )

    for result in results:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())