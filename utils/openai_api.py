import openai
import os
from time import sleep
import json
import requests

openai.api_key = ''

"""
def requestChatCompletion(text, t=0, n=1, model='gpt-3.5-turbo'):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text}
                ],
                temperature=t,
                n=n
            )

            if n == 1:
                message = response['choices'][0]['message']
                return message['content'], response
            else:
                message = [r['message']['content'] for r in response['choices']]
                return message, response
        except BaseException as e:
            print(e)
            sleep(1)
"""


api_url = "http://123.129.219.111:3000/v1/chat/completions"
api_key = "sk-Iar1GdsxnnSziiS9wSy3pVkUOmOc0iKVsTcdfYOrDQZ3RIKs"

def requestChatCompletion(messages, model='gpt-4o', finish_try=3, temperature=0, n=1):
    while finish_try > 0:
        try:
            payload = json.dumps({
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a data science expert."},
                    {"role": "user", "content": messages}
                ],
                "temperature": temperature,
                "n": n
            })

            headers = {
                'Authorization': f"Bearer {api_key}",
                'Content-Type': 'application/json',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
            }

            response = requests.post(api_url, headers=headers, data=payload)
            print("response ", response)
            print("response.status_code", response.status_code)

            if response.status_code == 200:
                response_data = response.json()
                if n == 1:
                    return response_data['choices'][0]['message']['content'], response_data
                else:
                    return [choice['message']['content'] for choice in response_data['choices']], response_data

        except Exception as e:
            print("Error:", e)
            finish_try -= 1
            continue
