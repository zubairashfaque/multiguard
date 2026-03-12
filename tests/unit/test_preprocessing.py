"""Tests for data preprocessing."""

import pytest


class TestTextPreprocessor:
    """Test suite for TextPreprocessor."""

    def test_strip_whitespace(self):
        """Test text cleaning strips whitespace."""
        from src.data.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor()
        assert preprocessor("  hello world  ") == "hello world"

    def test_empty_string(self):
        """Test text cleaning handles empty strings."""
        from src.data.preprocessing import TextPreprocessor

        preprocessor = TextPreprocessor()
        assert preprocessor("   ") == ""


class TestImagePreprocessor:
    """Test suite for ImagePreprocessor."""

    def test_valid_extensions(self):
        """Test validation accepts valid image extensions."""
        from src.data.preprocessing import ImagePreprocessor

        preprocessor = ImagePreprocessor()
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            # Only checks extension, not file existence in validate
            assert ext.lower() in preprocessor.VALID_EXTENSIONS

    def test_invalid_extension(self, tmp_path):
        """Test validation rejects invalid extensions."""
        from src.data.preprocessing import ImagePreprocessor

        preprocessor = ImagePreprocessor()
        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        assert preprocessor.validate(txt_file) is False
