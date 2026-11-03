# Day 4: DataLoader Preparation

Date: October 31, 2026

## Objective

Prepare tokenized SST-2 data in batches for RoBERTa.

## Configuration

- Batch size: 8
- Sequence length: 128
- Training data: shuffled
- Validation data: not shuffled

## Batch Contents

- `input_ids`: token IDs
- `attention_mask`: identifies real tokens and padding
- `labels`: sentiment labels

## Batch Shapes

- `input_ids`: `[8, 128]`
- `attention_mask`: `[8, 128]`
- `labels`: `[8]`

## Conclusion

The tokenized SST-2 data was successfully converted into PyTorch batches suitable for RoBERTa training.