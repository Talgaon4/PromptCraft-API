# tests/test_api.py

"""Comprehensive tests for the standardized PromptOptimizer API."""

import pytest
import tempfile
import shutil
from pathlib import Path

from prompt_optimizer.api import PromptOptimizer
from prompt_optimizer.response_objects import PromptResult, OptimizationResult, ValidationResult, OperationResult
from prompt_optimizer.exceptions import StorageError


class TestPromptOptimizerAPI:
    """Test the main PromptOptimizer API with standardized responses."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def optimizer(self, temp_dir):
        """Create a PromptOptimizer instance for testing."""
        return PromptOptimizer(
            storage_dir=temp_dir,
            optimization_threshold=3,  # Low threshold for testing
            strategy_name="simple_ai"
        )
    
    def test_register_prompt_success(self, optimizer):
        """Test successful prompt registration."""
        result = optimizer.register_prompt(
            text="Classify the sentiment: {input_text}",
            description="Sentiment classifier"
        )
        
        # Verify response type and structure
        assert isinstance(result, PromptResult)
        assert result.success is True
        assert result.prompt_id is not None
        assert result.prompt_text == "Classify the sentiment: {input_text}"
        assert result.version == 1
        assert result.message is not None
        assert result.timestamp is not None
        assert len(result.errors) == 0
        
        # Verify data field contains prompt info
        assert result.data is not None
        assert "id" in result.data
        assert "text" in result.data
    
    def test_register_prompt_validation_error(self, optimizer):
        """Test prompt registration with validation errors."""
        result = optimizer.register_prompt(text="")  # Empty text should fail
        
        assert isinstance(result, PromptResult)
        assert result.success is False
        assert result.prompt_id is None
        assert result.message is not None
        assert len(result.errors) > 0
        assert result.timestamp is not None
    
    def test_get_prompt_success(self, optimizer):
        """Test successful prompt retrieval."""
        # First create a prompt
        create_result = optimizer.register_prompt("Test prompt", "Test description")
        assert create_result.success
        prompt_id = create_result.prompt_id
        
        # Now get it
        result = optimizer.get_prompt(prompt_id)
        
        assert isinstance(result, PromptResult)
        assert result.success is True
        assert result.prompt_id == prompt_id
        assert result.prompt_text == "Test prompt"
        assert result.version == 1
        assert result.message is not None
        assert result.data is not None
    
    def test_get_prompt_not_found(self, optimizer):
        """Test prompt retrieval for non-existent prompt."""
        result = optimizer.get_prompt("non-existent-id")
        
        assert isinstance(result, PromptResult)
        assert result.success is False
        assert result.prompt_id is None
        assert result.message is not None
        assert "not found" in result.message.lower()
    
    def test_validate_prompt_id_success(self, optimizer):
        """Test prompt ID validation for existing prompt."""
        # Create a prompt first
        create_result = optimizer.register_prompt("Test prompt")
        prompt_id = create_result.prompt_id
        
        # Validate it
        result = optimizer.validate_prompt_id(prompt_id)
        
        assert isinstance(result, ValidationResult)
        assert result.success is True
        assert result.is_valid is True
        assert result.message is not None
        assert result.validation_details is not None
        assert result.validation_details["exists"] is True
    
    def test_validate_prompt_id_not_found(self, optimizer):
        """Test prompt ID validation for non-existent prompt."""
        result = optimizer.validate_prompt_id("non-existent-id")
        
        assert isinstance(result, ValidationResult)
        assert result.success is False
        assert result.is_valid is False
        assert result.message is not None
        assert result.validation_details is not None
        assert result.validation_details["exists"] is False
    
    def test_validate_prompt_id_invalid_format(self, optimizer):
        """Test prompt ID validation with invalid format."""
        result = optimizer.validate_prompt_id("")  # Empty ID
        
        assert isinstance(result, ValidationResult)
        assert result.success is False
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_record_prompt_use_success(self, optimizer):
        """Test successful prompt usage recording."""
        # Create a prompt first
        create_result = optimizer.register_prompt("Summarize: {text}")
        prompt_id = create_result.prompt_id
        
        # Record usage
        result = optimizer.record_prompt_use(
            prompt_id=prompt_id,
            formatted_text="Summarize: This is a test document.",
            context={"source": "test"}
        )
        
        assert isinstance(result, OperationResult)
        assert result.success is True
        assert result.data is not None
        assert "instance_id" in result.data
        assert "prompt_id" in result.data
        assert result.data["prompt_id"] == prompt_id
        assert result.message is not None
    
    def test_record_prompt_use_invalid_prompt(self, optimizer):
        """Test prompt usage recording with invalid prompt ID."""
        result = optimizer.record_prompt_use(
            prompt_id="invalid-id",
            formatted_text="Test text"
        )
        
        assert isinstance(result, OperationResult)
        assert result.success is False
        assert result.message is not None
        assert len(result.errors) > 0
    
    def test_record_response_success(self, optimizer):
        """Test successful response recording."""
        # Setup: create prompt and record usage
        create_result = optimizer.register_prompt("Test prompt")
        prompt_id = create_result.prompt_id
        
        usage_result = optimizer.record_prompt_use(prompt_id, "Formatted prompt")
        instance_id = usage_result.data["instance_id"]
        
        # Record response
        result = optimizer.record_response(
            prompt_instance_id=instance_id,
            content="This is the AI response",
            metadata={"model": "test-model"}
        )
        
        assert isinstance(result, OperationResult)
        assert result.success is True
        assert result.data is not None
        assert "response_id" in result.data
        assert "prompt_instance_id" in result.data
        assert result.data["prompt_instance_id"] == instance_id
    
    def test_record_response_invalid_instance(self, optimizer):
        """Test response recording with invalid instance ID."""
        result = optimizer.record_response(
            prompt_instance_id="invalid-instance",
            content="Test response"
        )
        
        assert isinstance(result, OperationResult)
        assert result.success is False
        assert result.message is not None
        assert len(result.errors) > 0
    
    def test_record_feedback_success(self, optimizer):
        """Test successful feedback recording."""
        # Setup: create prompt, record usage and response
        create_result = optimizer.register_prompt("Test prompt")
        prompt_id = create_result.prompt_id
        
        usage_result = optimizer.record_prompt_use(prompt_id, "Formatted prompt")
        instance_id = usage_result.data["instance_id"]
        
        response_result = optimizer.record_response(instance_id, "AI response")
        response_id = response_result.data["response_id"]
        
        # Record feedback
        result = optimizer.record_feedback(
            response_id=response_id,
            is_positive=True,
            score=0.8,
            comments="Great response!"
        )
        
        assert isinstance(result, OperationResult)
        assert result.success is True
        assert result.data is not None
        assert "feedback_id" in result.data
        assert "response_id" in result.data
        assert result.data["response_id"] == response_id
    
    def test_record_feedback_invalid_score(self, optimizer):
        """Test feedback recording with invalid score."""
        result = optimizer.record_feedback(
            response_id="fake-response-id",
            is_positive=True,
            score=2.0  # Invalid - should be 0-1
        )
        
        assert isinstance(result, OperationResult)
        assert result.success is False
        assert result.message is not None
        assert len(result.errors) > 0
        assert any("score" in error.lower() for error in result.errors)
    
    def test_get_optimization_stats_success(self, optimizer):
        """Test getting optimization statistics."""
        # Create a prompt
        create_result = optimizer.register_prompt("Test prompt")
        prompt_id = create_result.prompt_id
        
        # Get stats
        result = optimizer.get_optimization_stats(prompt_id)
        
        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert result.original_prompt_id == prompt_id
        assert result.readiness_info is not None
        assert "feedback_count" in result.readiness_info
        assert "threshold" in result.readiness_info
        assert "is_ready" in result.readiness_info
        assert result.data is not None
    
    def test_get_optimization_stats_invalid_prompt(self, optimizer):
        """Test getting stats for invalid prompt."""
        result = optimizer.get_optimization_stats("invalid-prompt-id")
        
        assert isinstance(result, OptimizationResult)
        assert result.success is False
        assert result.message is not None
        assert len(result.errors) > 0
    
    def test_optimize_prompt_not_ready(self, optimizer):
        """Test optimization when prompt is not ready."""
        # Create a prompt without enough feedback
        create_result = optimizer.register_prompt("Test prompt")
        prompt_id = create_result.prompt_id
        
        result = optimizer.optimize_prompt(prompt_id)
        
        assert isinstance(result, OptimizationResult)
        assert result.success is False
        assert result.original_prompt_id == prompt_id
        assert result.optimization_applied is False
        assert result.readiness_info is not None
        assert "not enough" in result.message.lower() or "not ready" in result.message.lower()
    
    def test_optimize_prompt_force(self, optimizer):
        """Test forced optimization."""
        # Create a prompt
        create_result = optimizer.register_prompt("Test prompt for optimization")
        prompt_id = create_result.prompt_id
        
        # Force optimization
        result = optimizer.optimize_prompt(prompt_id, force=True)
        
        assert isinstance(result, OptimizationResult)
        assert result.original_prompt_id == prompt_id
        
        # Result could be success or failure depending on strategy behavior
        # But should always have consistent structure
        if result.success:
            # If successful, check for optimization data
            if result.optimization_applied:
                assert result.new_prompt_id is not None
            else:
                assert result.data is not None or result.improvement_reason is not None
        else:
            assert result.message is not None
    
    def test_get_config_info(self, optimizer):
        """Test getting configuration information."""
        result = optimizer.get_config_info()
        
        assert isinstance(result, OperationResult)
        assert result.success is True
        assert result.data is not None
        
        config_data = result.data
        assert "optimization_threshold" in config_data
        assert "strategy" in config_data
        assert "storage_dir" in config_data
        assert "auto_apply" in config_data
    
    def test_full_workflow(self, optimizer):
        """Test a complete workflow from prompt creation to optimization."""
        # 1. Create prompt
        create_result = optimizer.register_prompt(
            text="Classify sentiment: {text}",
            description="Sentiment classifier"
        )
        assert create_result.success
        prompt_id = create_result.prompt_id
        
        # 2. Validate prompt
        validation_result = optimizer.validate_prompt_id(prompt_id)
        assert validation_result.success and validation_result.is_valid
        
        # 3. Record some usage and responses with feedback
        for i in range(3):  # Create enough feedback to potentially trigger optimization
            # Record usage
            usage_result = optimizer.record_prompt_use(
                prompt_id=prompt_id,
                formatted_text=f"Classify sentiment: Test text {i}"
            )
            assert usage_result.success
            instance_id = usage_result.data["instance_id"]
            
            # Record response
            response_result = optimizer.record_response(
                prompt_instance_id=instance_id,
                content="positive" if i % 2 == 0 else "negative"
            )
            assert response_result.success
            response_id = response_result.data["response_id"]
            
            # Record feedback
            feedback_result = optimizer.record_feedback(
                response_id=response_id,
                is_positive=True,
                score=0.8,
                comments=f"Test feedback {i}"
            )
            assert feedback_result.success
        
        # 4. Check optimization stats
        stats_result = optimizer.get_optimization_stats(prompt_id)
        assert stats_result.success
        assert stats_result.readiness_info["feedback_count"] == 3
        
        # 5. Try optimization (might not be ready but should return consistent response)
        opt_result = optimizer.optimize_prompt(prompt_id, force=True)
        assert isinstance(opt_result, OptimizationResult)
        assert opt_result.original_prompt_id == prompt_id
        
        # 6. Get config info
        config_result = optimizer.get_config_info()
        assert config_result.success


class TestResponseObjectConsistency:
    """Test that all response objects have consistent structure."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for consistency tests."""
        temp_dir = tempfile.mkdtemp()
        optimizer = PromptOptimizer(storage_dir=temp_dir)
        yield optimizer
        shutil.rmtree(temp_dir)
    
    def test_all_responses_have_base_fields(self, optimizer):
        """Test that all response objects have required base fields."""
        # Test different response types
        responses = []
        
        # PromptResult
        responses.append(optimizer.register_prompt("Test"))
        responses.append(optimizer.get_prompt("invalid-id"))
        
        # ValidationResult  
        responses.append(optimizer.validate_prompt_id("invalid-id"))
        
        # OperationResult
        responses.append(optimizer.record_prompt_use("invalid", "text"))
        responses.append(optimizer.record_response("invalid", "content"))
        responses.append(optimizer.record_feedback("invalid", True))
        responses.append(optimizer.get_config_info())
        
        # OptimizationResult
        responses.append(optimizer.get_optimization_stats("invalid-id"))
        responses.append(optimizer.optimize_prompt("invalid-id"))
        
        # Check that all responses have base fields
        required_fields = ['success', 'message', 'errors', 'timestamp']
        
        for response in responses:
            for field in required_fields:
                assert hasattr(response, field), f"Response {type(response)} missing field: {field}"
                
            # Check field types
            assert isinstance(response.success, bool)
            assert response.message is None or isinstance(response.message, str)
            assert isinstance(response.errors, list)
            assert response.timestamp is not None
    
    def test_error_responses_have_meaningful_messages(self, optimizer):
        """Test that error responses have helpful messages."""
        error_responses = [
            optimizer.register_prompt(""),  # Empty text
            optimizer.get_prompt("invalid-id"),  # Non-existent prompt
            optimizer.validate_prompt_id(""),  # Empty ID
            optimizer.record_prompt_use("invalid", ""),  # Empty text
            optimizer.record_feedback("invalid", True, score=2.0),  # Invalid score
        ]
        
        for response in error_responses:
            assert not response.success
            assert response.message is not None
            assert len(response.message) > 0
            # Should have specific error information
            assert "error" in response.message.lower() or "invalid" in response.message.lower() or "failed" in response.message.lower() or "not found" in response.message.lower()


class TestAPIRobustness:
    """Test API robustness and error handling."""
    
    def test_invalid_storage_directory(self):
        """Test handling of invalid storage directory."""
        # Try to create optimizer with invalid storage path
        with pytest.raises(StorageError):
            PromptOptimizer(storage_dir="/invalid/path/that/does/not/exist")
    
    def test_concurrent_operations(self):
        """Test that API handles concurrent operations safely."""
        import tempfile
        import threading
        
        temp_dir = tempfile.mkdtemp()
        optimizer = PromptOptimizer(storage_dir=temp_dir)
        
        results = []
        
        def create_prompt(i):
            result = optimizer.register_prompt(f"Test prompt {i}")
            results.append(result)
        
        # Create multiple prompts concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_prompt, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        assert len(results) == 5
        for result in results:
            assert result.success
            assert result.prompt_id is not None
        
        # All prompt IDs should be unique
        prompt_ids = [r.prompt_id for r in results]
        assert len(set(prompt_ids)) == 5
        
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__])