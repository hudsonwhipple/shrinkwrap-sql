"""Shared fixtures for the weight-edit tests. Nothing here needs the real model or the Spider download."""

import json

import mlx.core as mx
import pytest

TINY_CONFIG = {
    "model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
    "vocab_size": 97, "max_position_embeddings": 256, "tie_word_embeddings": True, "rope_theta": 10000.0,
}


@pytest.fixture
def tiny_config():
    """The tiny model's config.json, as a fresh dict."""
    return dict(TINY_CONFIG)


@pytest.fixture
def tiny_model():
    """A randomly initialized 2-block Qwen3 (hidden size 64, 4 heads sharing 2 key/value heads)."""
    from mlx_lm.models import qwen3

    def make(seed: int = 0):
        mx.random.seed(seed)
        model = qwen3.Model(qwen3.ModelArgs(**TINY_CONFIG))
        mx.eval(model.parameters())
        return model

    return make


@pytest.fixture
def fake_spider(tmp_path):
    """A Spider folder with two training databases, one of them held out of SFT, and one dev database."""
    root = tmp_path / "spider_data"
    root.mkdir()
    train = [
        {"db_id": "seen", "question": "How many singers?", "query": "SELECT count(*) FROM singer"},
        {"db_id": "held", "question": "How many dorms?", "query": "SELECT count(*) FROM dorm"},
        {"db_id": "seen", "question": "List the names.", "query": "SELECT name FROM singer"},
        {"db_id": "held", "question": "List the dorm names.", "query": "SELECT dorm_name FROM dorm"},
    ]
    dev = [{"db_id": "unseen", "question": "How many dorms?", "query": "SELECT count(*) FROM dorm"}]
    (root / "train_spider.json").write_text(json.dumps(train))
    (root / "dev.json").write_text(json.dumps(dev))
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"valid_dbs": ["held"]}))
    return root, meta
