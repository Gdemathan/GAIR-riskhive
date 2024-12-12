"""
Generate the submission file given a function that is able to generate 1 answer to 1 question
"""

from dataclasses import dataclass
import pandas as pd

ALLOWED_ANSWER = ['a','b','c','d']

@dataclass
class SubmissionBase:
    questions_df:pd.DataFrame

    _submission:pd.DataFrame = None

    def get_1_answer(self, q:str)->str:
        """This function should be implemented for all tries of prompt engineering"""
        raise NotImplementedError('The function should be implemented specifically in a subclass')
    
    def _get_submission(self)->pd.DataFrame:        
        def check_answer(a:str)->str:
            assert a in ALLOWED_ANSWER, f'The answer given "{a}" is not acceptable, it should be in {ALLOWED_ANSWER}'
            return a

        df = self.questions_df.copy()
        for i in range(1,6):
            df[f'predition_{i}'] = df['question'].apply(lambda q:check_answer(self.get_1_answer(q)))
        return df.drop('question',axis=1)
    
    def get_submission(self)->pd.DataFrame:
        if self._submission is None:
            self._submission = self._get_submission()
        return self._submission

    def score(self):
        from kaggle_score import get_score
        return get_score(self.get_submission())



if __name__=='__main__':
    class Example1(SubmissionBase):
        def get_1_answer(self,q):
            """In this example, the model gives always b.
            For futures improvements of our prompt engineering skills, we only have this function to define.
            """
            return 'b'

    questions = pd.read_csv('test.csv')
    sol = Example1(questions)
    print(sol.get_submission())
    sol.score()