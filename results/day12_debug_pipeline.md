# Day 12: Debug Data Pipeline

The data pipeline was refactored into `src/data.py`.

With debug mode enabled:

- Training examples: 128
- Validation examples: 128
- Training batches: 16
- Validation batches: 16


Debug validation accuracy: 47.66%
Debug validation F1: 0.0

This allows fast development without processing the full dataset.