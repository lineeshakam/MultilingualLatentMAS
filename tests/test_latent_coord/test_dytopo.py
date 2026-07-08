"""Unit tests for DyTopoBaseline (mechanism-level, CPU-only)."""

import pytest
import torch


def test_dytopo_topology_shape_and_no_self_loops():
    from latent_coordination.baselines.dytopo import DyTopoBaseline
    dt = DyTopoBaseline()
    states = {"a": torch.randn(16), "b": torch.randn(16), "c": torch.randn(16)}
    adj = dt.compute_topology(states, round_idx=0, threshold=0.5)
    assert adj.shape == (3, 3)
    assert torch.all(adj.diagonal() == 0)


def test_dytopo_identical_states_connect():
    from latent_coordination.baselines.dytopo import DyTopoBaseline
    dt = DyTopoBaseline()
    v = torch.randn(8)
    states = {"a": v, "b": v.clone()}
    adj = dt.compute_topology(states, threshold=0.5)
    assert adj[0, 1].item() == 1.0
    assert adj[1, 0].item() == 1.0


def test_dytopo_orthogonal_states_disconnect():
    from latent_coordination.baselines.dytopo import DyTopoBaseline
    dt = DyTopoBaseline()
    states = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}
    adj = dt.compute_topology(states, threshold=0.5)
    assert adj[0, 1].item() == 0.0


def test_dytopo_empty_states_raises():
    from latent_coordination.baselines.dytopo import DyTopoBaseline
    dt = DyTopoBaseline()
    with pytest.raises(ValueError):
        dt.compute_topology({})


def test_dytopo_diversity_stats_tracks_rounds():
    from latent_coordination.baselines.dytopo import DyTopoBaseline
    dt = DyTopoBaseline()
    torch.manual_seed(0)
    for r in range(3):
        dt.compute_topology({"a": torch.randn(8), "b": torch.randn(8)}, round_idx=r)
    stats = dt.topology_diversity_stats()
    assert stats["n_rounds"] == 3
    assert stats["mean_edge_count"] >= 0.0


def test_dytopo_decay_lowers_threshold_over_rounds():
    from latent_coordination.baselines.dytopo import DyTopoBaseline
    dt = DyTopoBaseline()
    # Fixed states with moderate similarity; a decaying threshold should only
    # ever make it easier (or equal) to connect as round_idx grows.
    a, b = torch.tensor([1.0, 0.3]), torch.tensor([1.0, 0.0])
    adj_early = dt.compute_topology({"a": a, "b": b}, round_idx=0, threshold=0.99, decay=0.5)
    adj_late = dt.compute_topology({"a": a, "b": b}, round_idx=5, threshold=0.99, decay=0.5)
    assert adj_late[0, 1].item() >= adj_early[0, 1].item()
