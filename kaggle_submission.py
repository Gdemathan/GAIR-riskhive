from dataclasses import dataclass
import pandas as pd
from src.utils import logger
from openai import OpenAI
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Literal


ALLOWED_ANSWER = ["a", "b", "c", "d"]


class Choice(BaseModel):
    choice: Literal["a", "b", "c", "d"]


@dataclass
class SubmissionBase(ABC):
    """
    The base class to build a submission for the Kaggle competition.
    Allows to ask questions to the IA assistant and get the answers, and save them
    in a CSV file.
    """

    questions_df: pd.DataFrame
    openai_client: OpenAI
    print_advancement = True

    @abstractmethod
    def get_1_answer(self, q: str) -> str:
        """
        Core function that returns a single answer to a question.
        """
        pass

    def _forcibly_extract_answer(self, q: str) -> str:
        """
        Extracts the answer from the question.
        """
        return (
            self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": q},
                    {"role": "user", "content": "Ok now give me the answer"},
                ],
                response_format=Choice,
            )
            .choices[0]
            .message.parsed.choice
        )

    def _get_submission(self, fake_multiple_attempts=False) -> pd.DataFrame:
        def check_answer(a: str, prediction_i, question_i) -> str:
            try:
                assert a in ALLOWED_ANSWER, f"Answer {a} is not in {ALLOWED_ANSWER}"
            except Exception:
                answer = self._forcibly_extract_answer(a)
                if self.print_advancement:
                    logger.info(
                        f" --> Prediction {prediction_i} for question {question_i} : {a}"
                    )
                return answer
            if self.print_advancement:
                logger.info(
                    f" --> Prediction {prediction_i} for question {question_i} : {a}"
                )
            return a

        df = self.questions_df.copy()
        for i in range(1, 6):
            if i != 1 and fake_multiple_attempts:
                df[f"prediction_{i}"] = df["prediction_1"]
                continue
            df[f"prediction_{i}"] = df.apply(
                lambda row: check_answer(
                    self.get_1_answer(row["question"]), i, row["question_id"]
                ),
                axis=1,
            )
        return df.drop("question", axis=1)

    def get_submission(
        self,
        save_path: str | None = "generated/submission.csv",
        fake_multiple_attempts=False,
    ) -> pd.DataFrame:
        submission = self._get_submission(fake_multiple_attempts=fake_multiple_attempts)
        if save_path:
            submission.to_csv(save_path, index=False)
        return submission


class SequentialQuestions(SubmissionBase):
    def get_1_answer(self, q: str, return_log=False) -> str:
        """This function should be implemented for all tries of prompt engineering.
        The log should then be the list of messages ('autor','message')"""
        raise NotImplementedError(
            "The function should be implemented specifically in a subclass"
        )

    def _ask_questions_in_a_row(self, context: str, questions: list) -> list:
        """The goal is to ask several questions in a row to the IA assistant to create a conversation.

        Input:
            prompt : the initial context for the assistant
            questions : the list of questions that will be given to the assitant

        Returns:
            the list of messages in the chat with the assistant
        """

        messages = [
            {"role": "system", "content": context},
        ]

        for q in questions:
            messages += [{"role": "user", "content": q}]

            chat = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            r = chat.choices[0].message.content

            messages += [{"role": "assistant", "content": r}]
        messages = [(f"{i}-" + m["role"], m["content"]) for i, m in enumerate(messages)]
        return messages

    def test(self, n=None) -> pd.DataFrame:
        df = pd.read_csv("data/train.csv")
        if isinstance(n, int):
            df = df[:n]

        def get_answer(row):
            logs = self.get_1_answer(row["question"], return_log=True)
            return pd.Series({k: v for k, v in logs})

        df_logs = df.apply(lambda row: get_answer(row), axis=1)
        df_logs = pd.DataFrame.from_dict(df_logs)
        df = pd.concat([df, df_logs], axis=1)
        return df


def test_submission(
    submitter: SubmissionBase, fake_multiple_attempts: bool = False
) -> float:
    initial_questions_df = submitter.questions_df.copy()
    submitter.questions_df = pd.read_csv("data/train.csv")
    answer_df = submitter.get_submission(
        fake_multiple_attempts=fake_multiple_attempts, save_path=None
    )
    score = answer_df.apply(
        lambda row: sum(
            [1 if row[f"prediction_{i}"] == row["answer"] else 0 for i in range(1, 6)]
        )
        / 5,
        axis=1,
    ).mean()
    submitter.questions_df = initial_questions_df

    logger.info("--------------------")
    logger.info(f"Score : {score} for model {submitter.__class__.__name__}")
    logger.info("--------------------")

    return score


if __name__ == "__main__":
    questions = pd.read_csv("data/test.csv")

    ai = SequentialQuestions(questions)
    ai.print_advancement = False
    r = ai._ask_questions_in_a_row(
        context="Hello World",
        questions=[
            "Quelle est la capitale de la France ?",
            "Combien y-a-t'il d'habitants dans cette ville ?",
        ],
    )
    [print(i) for i in r]
