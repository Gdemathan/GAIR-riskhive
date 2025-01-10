import numpy as np
from qdrant_client import QdrantClient
from tqdm import tqdm


if __name__ == "__main__":
    from utils import read_json
    from client import openai_client
else:
    from src.utils import read_json
    from src.client import openai_client, qdrant_client


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space.
    It is a value between -1 and 1, where:
    - 1 indicates that the vectors are identical in direction,
    - 0 indicates that the vectors are orthogonal,
    - -1 indicates that the vectors are diametrically opposed.

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
    def search(self, question):
        pass


class QdrantRAG(RAG):
    DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    SPARSE_MODEL = "prithivida/Splade_PP_en_v1"

    def __init__(self, rag_path, qdrant_client: QdrantClient = qdrant_client):
        self.qdrant_client: QdrantClient = qdrant_client
        self.collection_name = "masterclass"

        self.qdrant_client.set_model(self.DENSE_MODEL)
        self.qdrant_client.set_sparse_model(self.SPARSE_MODEL)

        if not qdrant_client.collection_exists(self.collection_name):
            qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_client.get_fastembed_vector_params(),
                sparse_vectors_config=qdrant_client.get_fastembed_sparse_vector_params(),
            )
            documents = []

            rag_list = read_json(rag_path)

            for idx, str in enumerate(rag_list):
                documents.append(str)

            qdrant_client.add(
                collection_name=self.collection_name,
                documents=documents,
                # metadata=metadata,
                ids=tqdm(range(len(documents))),
            )

    def search(self, text: str, limit: int = 5):
        search_result = self.qdrant_client.query(
            collection_name=self.collection_name,
            query_text=text,
            query_filter=None,  # If you don't want any filters for now
            limit=limit,  # 5 the closest results
        )
        metadata = [hit.metadata for hit in search_result]
        return metadata


class HandMadeRAG(RAG):
    d = None

    def __init__(self, rag_path, client=openai_client, use_qdrant=True):
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
            self.d = {k: self._get_embedding(str(k)) for k in self.to_list()}
        return self.d

    def __str__(self):
        short_dict = {k[:50]: v[:3] for k, v in self.to_dict().items()}
        _str = "RAG object : {"
        for k, v in short_dict.items():
            _str += f"""\n '{k}' : {v}"""
        _str += """\n}\n"""

        return _str

    def search(self, question):
        q_embedding = self._get_embedding(question)
        cosine = {
            k: cosine_similarity(v, q_embedding) for k, v in self.to_dict().items()
        }
        return max(cosine, key=lambda k: abs(cosine[k]))


if __name__ == "__main__":
    rag = RAG("RAG.json")
    print(rag)
    print(rag.search("bonjour"))
