import os
from litellm import embedding
import logging

# Silence excessive logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

os.environ["GEMINI_API_KEY"] = "AIzaSyB0EbSEdDcfu3bgUzILq3p_mk4isMxpnQI"

def test_config(model, api_version=None):
    print(f"\n--- Testing: model={model}, api_version={api_version} ---")
    if api_version:
        os.environ["GEMINI_API_VERSION"] = api_version
    else:
        if "GEMINI_API_VERSION" in os.environ:
            del os.environ["GEMINI_API_VERSION"]
            
    try:
        response = embedding(
            model=model,
            input=["test"]
        )
        print(f"SUCCESS! Vector length: {len(response['data'][0]['embedding'])}")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    models_to_test = [
        "gemini/text-embedding-004",
        "gemini/embedding-001",
        "gemini/gemini-embedding-001",
        "gemini/models/text-embedding-004",
        "gemini/models/embedding-001",
    ]
    
    versions = ["v1beta", "v1"]
    
    results = []
    for model in models_to_test:
        for version in versions:
            if test_config(model, version):
                results.append((model, version))
                
    print("\n" + "="*30)
    print("FINAL SUCCESSFUL CONFIGS:")
    for res in results:
        print(f"Model: {res[0]}, Version: {res[1]}")
    print("="*30)
