import numpy as np
from qdrant_client import QdrantClient
from tqdm import tqdm
import os


if __name__ == "__main__":
    from utils import read_json, save_json, logger
    from client import openai_client
else:
    from src.utils import read_json, save_json, logger
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
            print("Populating database...")
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
                ids=tqdm(range(len(documents))),
            )

    def search(self, text: str, limit: int = 5):
        search_result = self.qdrant_client.query(
            collection_name=self.collection_name,
            query_text=text,
            query_filter=None,
            limit=limit,
        )
        doc = [hit.document for hit in search_result]
        return doc


SEED_PATH = "data/rag_db_seed.json"


class HandMadeRAG(RAG):
    d = None

    def __init__(self, db_path, openai_client=openai_client, restart_db=False):
        if restart_db and os.path.exists(db_path):
            os.remove(db_path)
        self.db_path = db_path
        self.db = []
        self.client = openai_client
        if not os.path.exists(self.db_path):
            self._populate_db(SEED_PATH, db_path)
        self.db = read_json(self.db_path)

    def push_to_db(self):
        save_json(self.db, self.db_path)

    def _populate_db(self, source_path: str, db_path: str):
        try:
            elements = read_json(source_path)
            logger.info("Populating database...")
            for i in tqdm(range(len(elements))):
                element = elements[i]
                embedding = self._get_embedding(element)
                entry = {"text": element, "embedding": list(embedding)}
                self.db.append(entry)
                save_json(self.db, db_path)
        except Exception as e:
            logger.error(f"Error while populating the database: {e}")
            os.remove(db_path)
            raise e

    def _get_embedding(self, text: str) -> np.ndarray:
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
        return self.db

    def __str__(self):
        short_dict = {k[:50]: v[:3] for k, v in self.to_dict().items()}
        _str = "RAG object : {"
        for k, v in short_dict.items():
            _str += f"""\n '{k}' : {v}"""
        _str += """\n}\n"""

        return _str

    def search(self, question, limit=1):
        question_embedding = self._get_embedding(question)
        return [
            el["text"]
            for el in sorted(
                self.db,
                key=lambda el: abs(
                    cosine_similarity(np.array(el["embedding"]), question_embedding)
                ),
                reverse=True,
            )[:limit]
        ]


if __name__ == "__main__":
    rag = RAG("RAG.json")
    print(rag)
    print(rag.search("bonjour"))
