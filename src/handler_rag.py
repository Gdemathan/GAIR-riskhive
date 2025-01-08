import numpy as np

if __name__ == "__main__":
    from utils import read_json
    from client import openai_client
else:
    from src.utils import read_json
    from src.client import openai_client


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space.
    It is a value between -1 and 1, where:
    - 1 indicates that the vectors are identical in direction,
    - 0 indicates that the vectors are orthogonal,
    - -1 indicates that the vectors are diametrically opposed.

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
    """
    if vec1.shape != vec2.shape:
        raise ValueError("Both vectors must have the same dimensions.")

    dot_product = np.dot(vec1, vec2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        raise ZeroDivisionError("One or both of the vectors is a zero vector.")

    return dot_product / (norm_vec1 * norm_vec2)


class RAG:
    d = None

    def __init__(self, rag_path, client=openai_client):
        self.rag_path = rag_path
        self.json = read_json(self.rag_path)

        self.client = client

    def _get_embedding(self, text: str) -> np.array:
        """
        Retrieves the embedding of a given text using a pre-defined OpenAI client.

        Parameters:
            text (str): The input string to embed.

        Returns:
            list: A list of floats representing the embedding.
        """
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
            encoding_format="float",
        )
        list_ = response.data[0].embedding
        return np.array(list_)

    def to_list(self):
        return self.json

    def to_dict(self) -> dict:
        if self.d is None:
            self.d = {k: self._get_embedding(k) for k in self.to_list()}
        return self.d

    def __str__(self):
        short_dict = {k[:50]: v[:3] for k, v in self.to_dict().items()}
        _str = "RAG object : {"
        for k, v in short_dict.items():
            _str += f"""\n '{k}' : {v}"""
        _str += """\n}\n"""

        return _str

    def best_fit(self, question):
        q_embedding = self._get_embedding(question)
        cosine = {
            k: cosine_similarity(v, q_embedding) for k, v in self.to_dict().items()
        }
        return max(cosine, key=lambda k: abs(cosine[k]))


if __name__ == "__main__":
    rag = RAG("RAG.json")
    print(rag)
    print(rag.best_fit("bonjour"))
