import os
import numpy as np

if __name__=='__main__':
    from utils import read_json
    from client import openai_client
else:
    from src.utils import read_json
    from src.client import openai_client

def cosine_similarity(vec1:np.array, vec2:np.array)->float:
    """
    Computes the cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space. 
    It is a value between -1 and 1, where:
    - 1 indicates that the vectors are identical in direction,
    - 0 indicates that the vectors are orthogonal,
    - -1 indicates that the vectors are diametrically opposed.

    The formula for cosine similarity is:
        cosine_similarity = dot(vec1, vec2) / (||vec1|| * ||vec2||)
    where:
        - dot(vec1, vec2) is the dot product of vec1 and vec2,
        - ||vec1|| is the Euclidean norm (magnitude) of vec1,
        - ||vec2|| is the Euclidean norm (magnitude) of vec2.

    Parameters:
    ----------
    vec1 : np.ndarray
        A 1-dimensional NumPy array representing the first vector. It must be non-empty.
    vec2 : np.ndarray
        A 1-dimensional NumPy array representing the second vector. It must have the same dimensions as `vec1`.

    Returns:
    -------
    float
        The cosine similarity value between `vec1` and `vec2`.

    Raises:
    ------
    ValueError
        If `vec1` and `vec2` are not 1-dimensional arrays or have different lengths.
    ZeroDivisionError
        If either `vec1` or `vec2` is a zero vector (i.e., all elements are zero).

    Examples:
    --------
    >>> vec1 = np.array([1, 2, 3])
    >>> vec2 = np.array([4, 5, 6])
    >>> cosine_similarity(vec1, vec2)
    0.9746318461970762

    >>> vec1 = np.array([1, 0, 0])
    >>> vec2 = np.array([0, 1, 0])
    >>> cosine_similarity(vec1, vec2)
    0.0
    """
    # Check if vectors are of the same size
    if vec1.shape != vec2.shape:
        raise ValueError("Both vectors must have the same dimensions.")
    
    # Compute the dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute the magnitudes of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Check for zero vectors to avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        raise ZeroDivisionError("One or both of the vectors is a zero vector.")

    # Return the cosine similarity
    return dot_product / (norm_vec1 * norm_vec2)


    # Example usage
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    print(f"Cosine similarity: {cosine_similarity(vector1, vector2)}")


class RAG:
    d = None

    def __init__(self, rag_path, client=openai_client):
        self.rag_path = rag_path
        self.json = read_json(self.rag_path)

        self.client = client

    def _get_embedding(self,text:str)->np.array:
        """
        Retrieves the embedding of a given text using a pre-defined OpenAI client.
        
        Parameters:
            text (str): The input string to embed.
        
        Returns:
            list: A list of floats representing the embedding.
        """
        response = self.client.embeddings.create(
            input=text,
            model='text-embedding-3-small',
            encoding_format='float',
        )
        list_ = response.data[0].embedding
        return np.array(list_)
        
        # Example usage
        text = "OpenAI provides state-of-the-art AI models."
        embedding = get_embedding(text)

        # Print the embedding vector
        print(f"Embedding for the text:\n{text}\n")
        print(embedding[:10])
        print(f"Embedding length: {len(embedding)}")

    def to_list(self):
        return self.json

    def to_dict(self)->dict:
        if self.d is None:
            self.d = {k:self._get_embedding(k) for k in self.to_list()}
        return self.d

    def __str__(self):
        short_dict = {k[:50]:v[:3] for k,v in self.to_dict().items()}
        _str = "RAG object : {"
        for k,v in short_dict.items():
            _str+= f"""\n '{k}' : {v}"""
        _str+="""\n}\n"""

        return _str

    def best_fit(self, question):
        q_embedding = self._get_embedding(question)
        cosine = {k:cosine_similarity(v,q_embedding) for k,v in self.to_dict().items()}
        #print(cosine)
        return max(cosine, key=lambda k: abs(cosine[k]))

if __name__=='__main__':
    rag = RAG('RAG.json')
    print(rag)
    print(rag.best_fit('bonjour'))