"""Unit tests for OptimalAgentSelectionBaseline (mechanism-level, CPU-only)."""

import pytest


def test_select_agents_picks_higher_utility_within_budget():
    from latent_coordination.baselines.optimal_agent_selection import OptimalAgentSelectionBaseline
    sel = OptimalAgentSelectionBaseline()
    candidates = {
        "a": {"utility": 0.5, "cost": 1.0},
        "b": {"utility": 0.9, "cost": 1.0},
        "c": {"utility": 0.3, "cost": 1.0},
    }
    result = sel.select_agents(candidates, budget=2.0)
    assert set(result["selected_agents"]) == {"a", "b"}
    assert result["total_utility"] == pytest.approx(1.4)
    assert result["total_cost"] == pytest.approx(2.0)


def test_select_agents_respects_zero_budget():
    from latent_coordination.baselines.optimal_agent_selection import OptimalAgentSelectionBaseline
    sel = OptimalAgentSelectionBaseline()
    candidates = {"a": {"utility": 1.0, "cost": 1.0}}
    result = sel.select_agents(candidates, budget=0.0)
    assert result["selected_agents"] == []
    assert result["total_utility"] == 0.0


def test_select_agents_missing_field_raises():
    from latent_coordination.baselines.optimal_agent_selection import OptimalAgentSelectionBaseline
    sel = OptimalAgentSelectionBaseline()
    with pytest.raises(ValueError):
        sel.select_agents({"a": {"utility": 1.0}}, budget=1.0)


def test_select_agents_negative_cost_raises():
    from latent_coordination.baselines.optimal_agent_selection import OptimalAgentSelectionBaseline
    sel = OptimalAgentSelectionBaseline()
    with pytest.raises(ValueError):
        sel.select_agents({"a": {"utility": 1.0, "cost": -1.0}}, budget=1.0)


def test_select_agents_too_many_candidates_raises():
    from latent_coordination.baselines.optimal_agent_selection import OptimalAgentSelectionBaseline
    sel = OptimalAgentSelectionBaseline()
    candidates = {f"a{i}": {"utility": 1.0, "cost": 1.0} for i in range(21)}
    with pytest.raises(ValueError):
        sel.select_agents(candidates, budget=100.0)


def test_selection_rationale_requires_prior_selection():
    from latent_coordination.baselines.optimal_agent_selection import OptimalAgentSelectionBaseline
    sel = OptimalAgentSelectionBaseline()
    with pytest.raises(RuntimeError):
        sel.selection_rationale()


def test_selection_rationale_matches_last_call():
    from latent_coordination.baselines.optimal_agent_selection import OptimalAgentSelectionBaseline
    sel = OptimalAgentSelectionBaseline()
    sel.select_agents({"a": {"utility": 1.0, "cost": 1.0}}, budget=1.0)
    rationale = sel.selection_rationale()
    assert rationale["selected_agents"] == ["a"]
    assert rationale["unselected_agents"] == []
