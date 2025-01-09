"""
This file was used solely for the purpose of testing the system, for the corresponding assignment.
It is not used in the final product.
"""

from openai import OpenAI
import pandas as pd
import os
from pydantic import BaseModel
from typing import Literal

import dotenv

dotenv.load_dotenv()


class FullReasoning(BaseModel):
    steps: list[str]
    final_answer: Literal["a", "b", "c", "d"]


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are a reliability expert. Respond with a, b, c, or d.
"""

DOUBT_PROMPT = """
I have a doubt. Are you totally sure ? Double-check your answer and explain briefly in 2 steps.
"""


def provide_answers(input_file: str):
    """
    Answers each question for the training dataset.
    """
    data = pd.read_csv(input_file)

    score = 0

    if "question" not in data.columns:
        raise ValueError("The input file must have a 'question' column.")

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
            max_completion_tokens=150,
        )

        first_answer = first_response.choices[0].message.content

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": first_answer},
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
        if answer.final_answer != row["answer"]:
            print("MISTAKE FOUND!")
            print(f"Question: {question}\n")
            print(f"First answer: {first_answer}")
            print(f"Final answer: {answer}")
            print(f"Correct answer: {row['answer']}\n")
        else:
            score += 1
        print("\n\n")
    print(f"Final score: {score}/{len(data)}")

    os.makedirs("generated", exist_ok=True)
    print("Predictions saved successfully.")


provide_answers("data/train.csv")
