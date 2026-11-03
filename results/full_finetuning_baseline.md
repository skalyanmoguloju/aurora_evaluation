# Full Fine-Tuning Baseline

## Configuration

- Model: RoBERTa-base
- Dataset: SST-2
- Training examples: 67,349
- Validation examples: 872
- Batch size: 8
- Epochs: 1
- Device: Apple MPS

## Results

- Training time: 1,400.53 seconds
- Training time: 23.34 minutes
- Additional MPS memory: 1,736.57 MB
- Validation loss before training: 0.7032
- Validation accuracy before training: 49.08%
- Validation F1 before training: 0.0
- Training loss: 0.2423
- Final validation loss: 0.2029
- Final validation accuracy: 92.78%
- Final validation F1: 93.05%

## Conclusion

Full fine-tuning successfully trained RoBERTa on SST-2. After one epoch, the model achieved 92.78% validation accuracy and 93.05% validation F1 score.