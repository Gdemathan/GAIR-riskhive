from openai import OpenAI
import os
from qdrant_client import QdrantClient

try:
    import dotenv

    dotenv.load_dotenv()
except Exception:
    pass

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"),
    api_key=os.environ.get("QDRANT_API_KEY"),
)
