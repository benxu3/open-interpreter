from openai import OpenAI

# init client and connect to localhost server
client = OpenAI(
    api_key="fake-api-key",
    base_url="http://localhost:8000/v0",  # change the default port if needed
)

stream = client.chat.completions.create(
    model="mock-gpt-model",
    messages=[
        {
            "role": "user",
            "content": "testing this openai compatible endpoint, no need to do anything except to identify yourself! ",
        }
    ],
    stream=True,
)

for chunk in stream:
    print(chunk)
