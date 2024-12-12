"""
Average accuracy over five predictions.
"""

import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np


class ParticipantVisibleError(Exception):
    # If you want an error message to be shown to participants, you must raise the error as a ParticipantVisibleError
    # All other errors will only be shown to the competition host. This helps prevent unintentional leakage of solution data.
    pass


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    >>> import pandas as pd
    >>> num_rows = 5
    >>> data = {'question_id': list(range(num_rows))}
    >>> submission = pd.DataFrame(data)
    >>> for i in range(1, 6):
    >>>     column_name = f'prediction_{i}'
    >>>     submission[column_name] = ['a']*num_rows
    >>> solution = pd.DataFrame(data)
    >>> solution['answer'] = ['a']*num_rows
    >>> _ = score(solution.copy(), submission.copy(), 'question_id')
    Mean accuracy: 1.0
    '''
    if row_id_column_name in solution.columns:
        del solution[row_id_column_name]
    else:
        raise ParticipantVisibleError(f"solution.csv must have a column named {row_id_column_name}.")

    if row_id_column_name in submission.columns:
        del submission[row_id_column_name]
    else:
        raise ParticipantVisibleError(f"submission.csv must have a column named {row_id_column_name}.")

    results = []
    if submission.shape[1] != 5:
        raise ParticipantVisibleError(f"submission.csv must have 5 columns containing the 5 predicitons, but has {submission.shape[1]} columns.")


    y_true = solution['answer']
    for i in range(5):
        y_pred = submission.iloc[:, i]
        result = accuracy_score(y_true, y_pred)
        results.append(result)

    mean_acc = np.mean(results)

    print(f'Mean accuracy: {mean_acc}')

    return mean_acc

def get_score(submission:pd.DataFrame)->float:
    """
    Function used to check if a submission has the right formating.
    """
    num_rows = len(submission)
    solution_example = {'question_id': list(range(num_rows))}
    solution_example = pd.DataFrame(solution_example)
    solution_example['answer'] = ['a']*num_rows
    return score(solution_example.copy(), submission.copy(), 'question_id')


if __name__=='__main__':
    import pandas as pd
    num_rows = 5
    data = {'question_id': list(range(num_rows))}
    
    submission = pd.DataFrame(data)

    for i in range(1, 6):
        column_name = f'prediction_{i}'
        submission[column_name] = ['a']*num_rows
    
    solution = pd.DataFrame(data)
    solution['answer'] = ['a']*num_rows
    _ = score(solution.copy(), submission.copy(), 'question_id')

    _ = get_score(pd.read_csv('sample_submission.csv'))