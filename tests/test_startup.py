"""后端启动预热测试。"""

import asyncio
from unittest.mock import MagicMock, patch

from app import main


def test_startup_warms_shared_sparse_encoder():
    retrieval_service = MagicMock()

    with (
        patch.object(main, "init_db"),
        patch.object(main, "get_retrieval_service", return_value=retrieval_service),
    ):
        asyncio.run(main.startup_event())

    retrieval_service.hybrid_engine.sparse_encoder.warmup.assert_called_once()
