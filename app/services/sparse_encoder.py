"""共享、可持久化的财报 BM25 编码器。"""

import logging
from pathlib import Path
from typing import Iterable

import jieba
from pymilvus.model.sparse import BM25EmbeddingFunction

from app.core.config import settings

logger = logging.getLogger(__name__)


class SparseEncoderUnavailable(RuntimeError):
    """BM25 模型当前不可用于查询。"""


class PersistentBM25Encoder:
    """封装 PyMilvus BM25，并保证入库和查询复用同一词表。"""

    def __init__(self, model_path: str | Path | None = None):
        default_path = Path(settings.PROCESSED_DATA_PATH) / "finance_bm25.json"
        self.model_path = Path(model_path) if model_path is not None else default_path
        self.dirty_path = self.model_path.with_suffix(".dirty")
        self._model = None
        self._loaded_mtime_ns: int | None = None

    def fit(self, texts: Iterable[str]) -> None:
        corpus = list(texts)
        if not corpus:
            raise ValueError("BM25 corpus must not be empty")

        model = BM25EmbeddingFunction(analyzer=jieba.lcut)
        model.fit(corpus)
        self._model = model
        # None 表示当前模型来自内存中的最新 fit，而非磁盘文件。
        self._loaded_mtime_ns = None

    def save(self, path: str | Path | None = None) -> None:
        if self._model is None:
            raise SparseEncoderUnavailable("BM25 model has not been fitted")

        target = Path(path) if path is not None else self.model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(target))
        if target == self.model_path:
            self._loaded_mtime_ns = target.stat().st_mtime_ns

    def load(self) -> bool:
        """从磁盘加载可用模型；缺失、损坏或 dirty 时返回 False。"""
        if self.is_dirty or not self.model_path.is_file():
            self._model = None
            self._loaded_mtime_ns = None
            return False

        try:
            model = BM25EmbeddingFunction(analyzer=jieba.lcut)
            model.load(str(self.model_path))
            self._model = model
            self._loaded_mtime_ns = self.model_path.stat().st_mtime_ns
            return True
        except Exception as exc:
            self._model = None
            self._loaded_mtime_ns = None
            logger.warning("BM25 模型加载失败，将使用 dense_only: %s", exc)
            return False

    @property
    def is_dirty(self) -> bool:
        return self.dirty_path.exists()

    def mark_dirty(self) -> None:
        self.dirty_path.parent.mkdir(parents=True, exist_ok=True)
        self.dirty_path.write_text("rebuild_required\n", encoding="utf-8")
        self._model = None
        self._loaded_mtime_ns = None

    def clear_dirty(self) -> None:
        self.dirty_path.unlink(missing_ok=True)

    def encode_documents(self, texts: Iterable[str]):
        self._ensure_model(reject_dirty=False)
        return self._model.encode_documents(list(texts))

    def encode_query(self, query: str):
        self._ensure_model(reject_dirty=True)
        return self._model.encode_queries([query])

    def warmup(self, query: str = "财务报告") -> int:
        """提前加载持久化模型和 Jieba 词典，返回预热查询的非零项数。"""
        return int(self.encode_query(query).nnz)

    def _ensure_model(self, reject_dirty: bool) -> None:
        if reject_dirty and self.is_dirty:
            raise SparseEncoderUnavailable("BM25 corpus changed and requires rebuild")

        # fit() 后尚未保存的模型可以直接用于本次入库编码。
        if self._model is not None and self._loaded_mtime_ns is None:
            return

        try:
            current_mtime = self.model_path.stat().st_mtime_ns
        except OSError:
            current_mtime = None

        if self._model is None or current_mtime != self._loaded_mtime_ns:
            if not self.load():
                raise SparseEncoderUnavailable("BM25 model is unavailable")
