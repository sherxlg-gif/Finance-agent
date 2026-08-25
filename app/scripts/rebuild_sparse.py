"""一次性重建财报 Child 的 BM25 向量。"""

from pymilvus import Collection, connections

from app.core.config import settings
from app.services.sparse_rebuild import rebuild_sparse_vectors


def main() -> None:
    connections.connect(
        alias="default",
        host=settings.MILVUS_HOST,
        port=settings.MILVUS_PORT,
    )
    collection = Collection(settings.COLLECTION_NAME)
    result = rebuild_sparse_vectors(collection)
    print(
        "BM25 rebuild complete: "
        f"documents={result['documents']} upserted={result['upserted']}"
    )


if __name__ == "__main__":
    main()
