import openai


def createChat(messages, model):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,
        temperature=0.7,
    )
    gpt_response_content = response.choices[0].message.content
    return gpt_response_content
