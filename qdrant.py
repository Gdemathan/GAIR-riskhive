from src.client import qdrant_client

# TODO Delete this file. For now this is a POC of how to access the Qdrant database

docs = [
    "Qdrant has Langchain integrations",
    "Qdrant also has Llama Index integrations",
    "I am the king",
    "Spain",
]

# qdrant_client.add(
#     collection_name="demo_collection",
#     documents=docs,
# )

print(qdrant_client.count(collection_name="demo_collection"))

search_result = qdrant_client.query(
    collection_name="demo_collection", query_text="Queen", limit=1
)

print(search_result[0].document)
