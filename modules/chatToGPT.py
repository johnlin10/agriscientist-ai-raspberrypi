import openai


def createChat(messages):
    selected = 2
    models = ["gpt-3.5-turbo-16k-0613", "gpt-4-turbo", "gpt-4-turbo-2024-04-09"]
    response = openai.chat.completions.create(
        model=models[selected],
        messages=messages,
        max_tokens=2048,
        temperature=0.7,
    )
    gpt_response_content = response.choices[0].message.content
    return gpt_response_content
