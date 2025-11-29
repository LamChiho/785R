"""Tests for the main module."""

from src.main import main


def test_main_returns_zero():
    """Test that main function returns 0."""
    result = main()
    assert result == 0
