"""Unit tests for KVCommBaseline (mechanism-level, CPU-only)."""

import pytest
import torch


def test_kvcomm_register_and_fuse_shape():
    from latent_coordination.baselines.kvcomm import KVCommBaseline
    kv = KVCommBaseline()
    kv.register_agent("a", num_heads=2, head_dim=16)
    kv.register_agent("b", num_heads=4, head_dim=8)
    sk = torch.randn(2, 5, 32)
    sv = torch.randn(2, 5, 32)
    rk = torch.randn(2, 3, 32)
    rv = torch.randn(2, 3, 32)
    fk, fv = kv.fuse("a", "b", sk, sv, rk, rv, layer_idx=0)
    assert fk.shape == (2, 3, 32)
    assert fv.shape == (2, 3, 32)


def test_kvcomm_fuse_unregistered_raises():
    from latent_coordination.baselines.kvcomm import KVCommBaseline
    kv = KVCommBaseline()
    kv.register_agent("a", num_heads=1, head_dim=8)
    with pytest.raises(KeyError):
        kv.fuse("a", "ghost", torch.randn(1, 1, 8), torch.randn(1, 1, 8), torch.randn(1, 1, 8), torch.randn(1, 1, 8))


def test_kvcomm_injection_hook_calls_fuse():
    from latent_coordination.baselines.kvcomm import KVCommBaseline
    kv = KVCommBaseline()
    kv.register_agent("a", num_heads=1, head_dim=8)
    kv.register_agent("b", num_heads=1, head_dim=8)
    sender_kv = (torch.randn(1, 2, 8), torch.randn(1, 2, 8))
    hook = kv.build_kv_injection_hook("a", "b", sender_kv_provider=lambda: sender_kv)
    rk = torch.randn(1, 1, 8)
    rv = torch.randn(1, 1, 8)
    fk, fv = hook(rk, rv)
    assert fk.shape == rk.shape and fv.shape == rv.shape
    stats = kv.communication_stats()
    assert stats["n_fuse_calls"] == 1


def test_kvcomm_communication_stats():
    from latent_coordination.baselines.kvcomm import KVCommBaseline
    kv = KVCommBaseline()
    kv.register_agent("a", num_heads=1, head_dim=4)
    kv.register_agent("b", num_heads=1, head_dim=4)
    stats = kv.communication_stats()
    assert stats["n_registered_agents"] == 2
    assert stats["n_projection_pairs"] == 2  # a->b and b->a
    assert stats["n_fuse_calls"] == 0
