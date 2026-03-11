import cognee
import asyncio
import os

# LLM
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["LLM_MODEL"] = "gemini/gemini-2.0-flash"
os.environ["LLM_API_KEY"] = "AIzaSyB0EbSEdDcfu3bgUzILq3p_mk4isMxpnQI"

# Embedding
os.environ["EMBEDDING_PROVIDER"] = "gemini"
os.environ["EMBEDDING_MODEL"] = "gemini/gemini-embedding-001"
os.environ["EMBEDDING_API_KEY"] = "AIzaSyB0EbSEdDcfu3bgUzILq3p_mk4isMxpnQI"

async def test():
    print("Adding text...")
    await cognee.add("Hello world")
    print("Cognifying...")
    await cognee.cognify()
    print("Searching...")
    results = await cognee.search("Hello")
    print("Results:", results)

if __name__ == "__main__":
    asyncio.run(test())
