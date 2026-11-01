# Day 6: RoBERTa Forward Pass

## Result

A batch of 8 tokenized reviews was passed through RoBERTa.

- Logits shape: `[8, 2]`
- Initial loss: `0.6977`

The logits shape confirms that the model produces two class scores for every review. The initial loss is expected because the model has not yet been fine-tuned on SST-2.