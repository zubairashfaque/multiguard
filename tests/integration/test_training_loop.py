"""Integration tests for the training loop."""

import pytest


@pytest.mark.integration
class TestTrainingLoop:
    """Test the full training loop with tiny data."""

    @pytest.mark.skip(reason="Requires model loading — implement after backbones are ready")
    def test_single_epoch_baseline(self, tiny_config):
        """Test one training epoch completes without errors."""
        pass

    @pytest.mark.skip(reason="Requires model loading — implement after backbones are ready")
    def test_checkpoint_save_load(self, tiny_config, tmp_path):
        """Test checkpoint saving and loading."""
        pass
