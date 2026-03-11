import os
from litellm import embedding

os.environ["GEMINI_API_KEY"] = "AIzaSyB0EbSEdDcfu3bgUzILq3p_mk4isMxpnQI"

def test_embedding(model_name):
    print(f"\nTesting model: {model_name}")
    try:
        response = embedding(
            model=model_name,
            input=["Hello world"]
        )
        print(f"Success! Vector length: {len(response['data'][0]['embedding'])}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Test standard names
    test_embedding("gemini/text-embedding-004")
    test_embedding("gemini/gemini-embedding-001")
    # Test with models/ prefix if needed
    test_embedding("gemini/models/text-embedding-004")
