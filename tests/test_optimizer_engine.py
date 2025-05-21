# tests/test_optimizer_engine.py

"""Tests for OptimizerEngine functionality."""

import pytest
from unittest.mock import Mock, MagicMock

from prompt_optimizer.core.models import Prompt
from prompt_optimizer.core.optimizer_engine import OptimizerEngine


@pytest.fixture
def optimizer_engine():
    """Create an OptimizerEngine with mocked dependencies."""
    prompt_manager = Mock()
    feedback_collector = Mock()
    
    # Configure the mocks
    prompt_manager.get_prompt.return_value = Prompt(
        id="test-prompt-id",
        text="Original prompt text"
    )
    
    prompt_manager.update_prompt.return_value = Prompt(
        id="optimized-prompt-id",
        text="Original prompt text [optimized]",
        version=2,
        parent_id="test-prompt-id"
    )
    
    feedback_collector.calculate_feedback_stats.return_value = {
        "prompt_id": "test-prompt-id",
        "total_feedback": 15,
        "positive_ratio": 0.8,
        "average_score": 0.75,
        "has_sufficient_data": True
    }
    
    return OptimizerEngine(
        prompt_manager=prompt_manager,
        feedback_collector=feedback_collector,
        optimization_threshold=10,
        auto_apply=False
    )


def test_check_optimization_readiness(optimizer_engine):
    """Test checking if a prompt is ready for optimization."""
    readiness = optimizer_engine.check_optimization_readiness("test-prompt-id")
    
    assert readiness["is_ready"] is True
    assert readiness["feedback_count"] == 15
    assert readiness["threshold"] == 10


def test_generate_optimization(optimizer_engine):
    """Test generating an optimization for a prompt."""
    optimized_text = optimizer_engine.generate_optimization("test-prompt-id")
    
    assert optimized_text == "Original prompt text [optimized]"
    optimizer_engine.prompt_manager.get_prompt.assert_called_once_with("test-prompt-id")


def test_apply_optimization(optimizer_engine):
    """Test applying an optimization to a prompt."""
    optimized_id = optimizer_engine.apply_optimization(
        "test-prompt-id",
        "Optimized text"
    )
    
    assert optimized_id == "optimized-prompt-id"
    optimizer_engine.prompt_manager.update_prompt.assert_called_once_with(
        prompt_id="test-prompt-id",
        text="Optimized text"
    )