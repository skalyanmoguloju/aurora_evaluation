# Day 3: SST-2 Tokenization

Date: October 29, 2026

## Tokenizer

We used the pretrained RoBERTa tokenizer, which uses Byte-Level Byte Pair Encoding (BPE).

## Tokenization Process

Each review was converted into:

- `input_ids`: numerical token IDs
- `attention_mask`: identifies real tokens and padding

## Configuration

- Maximum sequence length: 128
- Longer sentences: truncated
- Shorter sentences: padded
- Beginning token: `<s>`
- Ending token: `</s>`
- Padding token: `<pad>`

## Observation

The review `"hide new secretions from the parental units"` produced 11 meaningful token positions, including special tokens. The remaining positions were padding tokens with ID `1`.

## Conclusion

The raw SST-2 sentences were successfully converted into fixed-length numerical inputs suitable for RoBERTa.