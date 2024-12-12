"""
Generate the submission file given a function that is able to generate 1 answer to 1 question
"""

from dataclasses import dataclass
import pandas as pd

ALLOWED_ANSWER = ['a','b','c','d']

@dataclass
class SubmissionBase:
    questions_df:pd.DataFrame
    print_avancement = True

    _submission:pd.DataFrame = None

    def get_1_answer(self, q:str)->str:
        """This function should be implemented for all tries of prompt engineering"""
        raise NotImplementedError('The function should be implemented specifically in a subclass')
    
    def _get_submission(self)->pd.DataFrame:        
        def check_answer(a:str,prediction_i,question_i)->str:
            try:
                assert a in ALLOWED_ANSWER
            except:
                print(f' !! Error Detected : The answer given "{a}" is not acceptable, it should be in {ALLOWED_ANSWER}')
            if self.print_avancement:
                print(f' --> Prediction {prediction_i}, question {question_i} : {a}')
            return a

        df = self.questions_df.copy()
        for i in range(1,6):
            df[f'predition_{i}'] = df.apply(lambda row:check_answer(self.get_1_answer(row['question']),i,row['question_id']),axis=1)
        return df.drop('question',axis=1)
    
    def get_submission(self)->pd.DataFrame:
        if self._submission is None:
            self._submission = self._get_submission()
        return self._submission

    def score(self):
        from kaggle_score import get_score
        return get_score(self.get_submission())
    
    def submission_to_csv(self,fname='submission.csv'):
        df = self.get_submission()
        df.to_csv(fname,index=False)
        return df


class SequentialQuestions(SubmissionBase):

    def get_1_answer(self, q:str, return_log = False)->str:
        """This function should be implemented for all tries of prompt engineering.
        The log should then be the list of messages ('autor','message')"""
        raise NotImplementedError('The function should be implemented specifically in a subclass')

    def _ask_questions_in_a_row(self,context:str,questions:list[str])->list[dict[str:str]]:
        """The goal is to ask several questions in a row to the IA assistant to create a conversation.

        Input:
            prompt : the initial context for the assistant
            questions : the list of questions that will be given to the assitant

        Returns:
            the list of messages in the chat with the assistant        
        """
        from openai import OpenAI
        import os
        client = OpenAI(
            api_key=os.environ.get('OPENAI_API_KEY'),  # This is the default and can be omitted
        )

        messages = [
            {"role": "system", "content": context},
        ]

        for q in questions:
            messages += [{"role": "user", "content": q}]

            chat = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                #temperature=0.61,
                #max_completion_tokens=50,
            )
            r = chat.choices[0].message.content

            messages += [{"role": "assistant", "content": r}]
        messages = [(f'{i}-'+m['role'],m['content']) for i,m in enumerate(messages)]
        return messages

    def test(self, n=None)->pd.DataFrame:
        df = pd.read_csv('train.csv')
        if isinstance(n,int):
            df = df[:n]

        def get_answer(row):
            logs = self.get_1_answer(row['question'], return_log=True)
            if self.print_avancement:
                print(row['question_id'])
                for i in logs: print(i)
                print('')
            return pd.Series({k:v for k,v in logs})

        df_logs = df.apply(lambda row:get_answer(row),axis=1)
        df_logs = pd.DataFrame.from_dict(df_logs)
        df = pd.concat([df, df_logs], axis=1)
        return df

if __name__=='__main__':

    questions = pd.read_csv('test.csv')

    if False:
        class Example1(SubmissionBase):
            def get_1_answer(self,q):
                """In this example, the model gives always b.
                For futures improvements of our prompt engineering skills, we only have this function to define.
                """
                return 'b'

        
        sol = Example1(questions)
        print(sol.submission_to_csv())
        sol.score()
    
    if True:
        ai = SequentialQuestions(questions)
        r = ai._ask_questions_in_a_row(
            context='Hello World',
            questions=[
                'Quelle est la capitale de la France ?',
                'Combien y-a-t\'il d\'habitants dans cette ville ?'
            ]
        )
        [print(i) for i in r]