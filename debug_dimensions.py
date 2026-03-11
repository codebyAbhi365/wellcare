import os
import litellm
from litellm import embedding

os.environ["GEMINI_API_KEY"] = "AIzaSyB0EbSEdDcfu3bgUzILq3p_mk4isMxpnQI"
os.environ["GEMINI_API_VERSION"] = "v1"

def test_dimensions(dim):
    print(f"\n--- Testing dimensions={dim} ---")
    try:
        kwargs = {
            "model": "gemini/gemini-embedding-001",
            "input": ["test text"]
        }
        if dim is not None:
            kwargs["dimensions"] = dim
            
        response = embedding(**kwargs)
        print(f"SUCCESS! Vector length: {len(response['data'][0]['embedding'])}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_dimensions(None)
    test_dimensions(768)
    test_dimensions(3072)
    test_dimensions(4096)
