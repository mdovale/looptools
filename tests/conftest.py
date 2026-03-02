"""Pytest configuration and shared fixtures for looptools tests."""

import matplotlib

# Use non-interactive backend to avoid display issues during tests
matplotlib.use("Agg")
