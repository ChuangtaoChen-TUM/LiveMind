import time
from config import get_model

model = get_model("gpt-4o")

message = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Hi. Just reply hi."
    }
]

start_time = time.time()
response = model.chat_complete(message=message)
end_time = time.time()
time_taken = end_time - start_time
gen_tokens = model.token_count.completion_tokens
speed = gen_tokens / time_taken
print(response)
print(f"Total Generation Time: {time_taken:.2f} seconds")
print(f"Number of tokens generated: {gen_tokens}")
print(f"Generation speed: {speed:.2f} tokens per second")