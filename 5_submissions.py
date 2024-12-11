from openai import OpenAI
import pandas as pd
import os
from pydantic import BaseModel
from typing import Literal


class FullReasoning(BaseModel):
    steps: list[str]
    final_answer: Literal["a", "b", "c", "d"]


class DoubleCheckedAnswer(BaseModel):
    explanation: str
    final_answer: Literal["a", "b", "c", "d"]


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are a reliability expert. Respond with a, b, c, or d.
"""

DOUBT_PROMPT = """
I have a doubt. Are you totally sure ? Double-check your answer and explain briefly in 2 steps.
"""


def process_questions(input_file):
    data = pd.read_csv(input_file)

    if "question" not in data.columns:
        raise ValueError("The input file must have a 'question' column.")

    output_file = pd.DataFrame(
        columns=["question_id", "attempt_1", "attempt_2", "attempt_3", "attempt_4", "attempt_5"]
    )
    output_file["question_id"] = data["question_id"]

    for index, row in data.iterrows():
        print(f"Processing question number {index + 1}/{len(data)}...")
        question = row["question"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        first_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.61,
            max_completion_tokens=50,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": first_response.choices[0].message.content},
            {"role": "user", "content": DOUBT_PROMPT},
        ]

        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.61,
            max_completion_tokens=300,
            response_format=FullReasoning,
        )

        answer = response.choices[0].message.parsed
        for i in range(1, 6):
            output_file.loc[index, f"attempt_{i}"] = answer.final_answer

    # Save to the output file
    os.makedirs("generated", exist_ok=True)
    output_file.to_csv("generated/full_output.csv", index=False)
    print("Predictions saved successfully.")

process_questions("test.csv")
