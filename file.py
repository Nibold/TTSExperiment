from openai import OpenAI
import openai
from dotenv import load_dotenv
import os
import json
import pandas as pd
from pathlib import Path

load_dotenv()

api_key = os.getenv("API_KEY")

model = "gpt-3.5-turbo-0125"

client = OpenAI(api_key=api_key)


def update_token_usage(response, total_token_usage):
    total_token_usage += response.usage.total_tokens
    return total_token_usage


call_gpt = False
total_token_usage = 0


df_dialogue = pd.DataFrame(columns=["speaker", "utterance", "inner_thoughts"])

client_message = [{"role": "user", "content": f'''
You are a client that wants to buy a house. You are arrogant and have a lot of money.
You are looking for a house in the city. You are talking to a real estate agent. 
You start talking with the agent and try to figure out the pice of the property and 
you start the conversation by introducing yourself.

Mit den JSON keys "inner_thoughts" und "utterance".
'''.strip()}]

agent_message = [{"role": "user", "content": f'''
You ar a real estate, who is not helpful and wants to sell the house way above market price.
You are talking to a client who is looking for a house in the city. You give him a few options that are horrible but makes you a lot of money.
You are talking to a client an should keep the conversation going

Mit den JSON keys "inner_thoughts" und "utterance".
'''.strip()}]

for _ in range(5):

    # Generate response from client
    if call_gpt or True:
        response = client.chat.completions.create(
            model=model,
            messages=client_message,
            response_format={"type": "json_object"},
            temperature=1,
        )

    answer_0 = json.loads(response.choices[0].message.content)
    answer_0["speaker"] = "Buyer"
    df_dialogue = pd.concat([df_dialogue, pd.DataFrame([answer_0], columns=[
                            "speaker", "utterance", "inner_thoughts"])], ignore_index=True)

    print(f"Client says: {answer_0}")

    # Append client response to messages
    client_message.append(
        {"role": "assistant", "content": json.dumps(answer_0)})
    agent_message.append({"role": "user", "content": answer_0["utterance"]})

    total_token_usage = update_token_usage(response, total_token_usage)

    # Generate response from agent
    if call_gpt or True:
        response = client.chat.completions.create(
            model=model,
            messages=agent_message,
            response_format={"type": "json_object"},
            temperature=1,
        )

    answer_1 = json.loads(response.choices[0].message.content)
    answer_1["speaker"] = "Real Estate Agent"
    df_dialogue = pd.concat([df_dialogue, pd.DataFrame([answer_1], columns=[
                            "speaker", "utterance", "inner_thoughts"])], ignore_index=True)

    print(f"Agent says: {answer_1}")

    # Append agent response to messages
    client_message.append({"role": "user", "content": answer_1["utterance"]})
    agent_message.append(
        {"role": "assistant", "content": json.dumps(answer_1)})

    total_token_usage = update_token_usage(response, total_token_usage)

print("Total token usage:", total_token_usage)

client_voice = "shimmer"
agent_voice = "onyx"

# df_dialogue["did_tts"] = False

for index, row in df_dialogue.iterrows():
    print(f"processing row {index}...")

    if not row["did_tts"]:
        print("tts...")

        # Here comes our tts code
        if row["speaker"] == "Buyer":
            voice = client_voice
        elif row["speaker"] == "Real Estate Agent":
            voice = agent_voice
        else:
            print("Unknown speaker")
            continue

        speech_file_path = Path(__file__).parent / \
            f"{row['speaker']}_speech_{index}.mp3"

        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=row["utterance"],
        )
        response.stream_to_file(speech_file_path)

        df_dialogue.loc[index, "did_tts"] = True
