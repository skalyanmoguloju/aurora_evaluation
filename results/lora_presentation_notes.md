# LoRA Presentation Notes

## 1. Starting point: the SST-2 sentence

Example sentence:

```text
hide new secretions from the parental units
```

The RoBERTa tokenizer adds special tokens:

```text
<s> hide new secretions from the parental units </s>
```

Because the model uses a maximum sequence length of 128, padding tokens are added:

```text
<s> hide new secretions from the parental units </s> <pad> <pad> ...
```

The tokenizer converts these tokens into integer token IDs. The IDs are vocabulary indexes, not numerical meanings.

Each token ID is then converted into a vector containing 768 values. Therefore, the complete padded sentence becomes:

\[
X:128\times768
\]

This means:

- 128 token positions;
- 768 features for each token.

LoRA and AuroRA operate on these 768-dimensional hidden representations, not directly on token IDs.

---

## 2. Full fine-tuning baseline

The first model was the pretrained `roberta-base` model with a two-class sentiment classifier.

For one token, the hidden representation is:

\[
x\in\mathbb{R}^{768}
\]

A RoBERTa attention projection uses a weight matrix:

\[
W_0:768\times768
\]

This matrix maps 768 input features to 768 output features:

\[
(1\times768)(768\times768)
=
1\times768
\]

The original transformation is:

\[
h_0=W_0x
\]

During full fine-tuning, all model parameters are allowed to change:

- embeddings;
- attention weights;
- feed-forward weights;
- normalization parameters;
- classification head.

The update is:

\[
W_{\text{new}}
=
W_{\text{old}}
-
\eta\frac{\partial L}{\partial W}
\]

Full fine-tuning provides maximum flexibility, but it requires updating approximately 125 million parameters.

Our full fine-tuning result was:

- Validation accuracy: 92.78%
- Validation F1-score: 93.05%
- Training time: 1,400.53 seconds
- Approximately 125 million trainable parameters

---

## 3. LoRA motivation

LoRA keeps the original pretrained matrix frozen:

\[
W_0\text{ is frozen}
\]

Instead of directly updating every element of \(W_0\), LoRA learns a small correction using two smaller matrices.

For RoBERTa:

\[
W_0:768\times768
\]

LoRA uses:

\[
A:8\times768
\]

\[
B:768\times8
\]

The bottleneck is therefore:

\[
768\rightarrow8\rightarrow768
\]

This temporary bottleneck applies to the adapter update. It does not permanently compress or replace the token representation.

---

## 4. LoRA forward pass

For one token representation \(x\), the original frozen path is:

\[
W_0x
\]

The LoRA path is:

\[
B(Ax)
\]

The complete LoRA output is:

\[
h_{\text{LoRA}}
=
W_0x+sB(Ax)
\]

where \(s\) is the LoRA scaling factor.

The two paths are added together:

```text
Original frozen path:  x → W₀ → original output
LoRA trainable path:   x → A → B → update
                         ↓
                 original output + update
```

The original RoBERTa computation remains active. LoRA only learns an additional correction.

---

## 5. Why LoRA uses fewer parameters

The original matrix contains:

\[
768\times768=589{,}824
\]

parameters.

The LoRA matrices contain:

\[
(8\times768)+(768\times8)
=
12{,}288
\]

parameters.

Therefore, LoRA trains far fewer parameters than updating the complete \(768\times768\) matrix.

The same `A` and `B` matrices are shared across all tokens processed by one particular projection. LoRA does not create a separate pair of matrices for every token.

Each token still receives an individual update because each token has a different input vector:

\[
\Delta h_i=B(Ax_i)
\]

The matrices are shared, but the output update differs for each token.

---

## 6. LoRA adapters in our project

Our configuration was:

```python
target_modules=["query", "value"]
```

RoBERTa has 12 Transformer layers. Each layer has a query projection and a value projection.

Therefore, LoRA was added separately to:

- the query projection of every layer;
- the value projection of every layer.

Each targeted module has its own LoRA matrices.

The same matrices are shared across tokens inside one module, but different layers and different target modules have different adapter parameters.

---

## 7. LoRA scaling

LoRA uses:

\[
s=\frac{\alpha}{r}
\]

Our settings were:

```python
r=8
lora_alpha=16
```

Therefore:

\[
s=\frac{16}{8}=2
\]

The LoRA output is:

\[
h_{\text{LoRA}}
=
W_0x+2B(Ax)
\]

The scaling factor controls how strongly the learned adapter update affects the original RoBERTa output.

It does not change the dimensions of the matrices.

---

## 8. LoRA dropout

Our configuration used:

```python
lora_dropout=0.1
```

During training, approximately 10% of values in the LoRA path are randomly dropped.

The dropout affects the adapter path:

\[
x\rightarrow A\rightarrow\text{dropout}\rightarrow B
\]

The original RoBERTa path remains available.

Dropout is disabled during validation and testing. Its purpose is to reduce overfitting.

---

## 9. Classifier training

The LoRA configuration also included:

```python
modules_to_save=["classifier"]
```

This means:

- the original RoBERTa encoder remains frozen;
- LoRA matrices are trainable;
- the sentiment classification head is trainable.

The classifier receives the final sentence representation and produces two logits:

\[
[\text{negative score},\text{positive score}]
\]

The class with the larger score becomes the prediction.

---

## 10. LoRA result

Our full LoRA experiment used:

```text
Learning rate: 2e-5
Epochs: 1
```

Results:

- Validation loss: 0.19966
- Validation accuracy: 92.55%
- Validation F1-score: 92.77%
- Training time: 835.17 seconds
- Trainable parameters: 887,042
- Total parameters: 125,534,212
- Trainable percentage: 0.7066%
- Additional MPS memory: 13.20 MB

Compared with full fine-tuning, LoRA was slightly less accurate but used far fewer trainable parameters and trained faster.

---

## 11. AuroRA

AuroRA uses the same frozen RoBERTa path and the same low-rank bottleneck as LoRA.

The difference is that AuroRA inserts a nonlinear activation between \(A\) and \(B\).

LoRA uses:

\[
h_{\text{LoRA}}
=
W_0x+sB(Ax)
\]

AuroRA uses:

\[
h_{\text{AuroRA}}
=
W_0x+sB(\operatorname{GELU}(Ax))
\]

The flow is:

```text
LoRA:
x → A → B

AuroRA:
x → A → GELU → B
```

Both methods use:

\[
768\rightarrow8\rightarrow768
\]

The bottleneck is not the difference between them. The difference is the GELU activation.

---

## 12. Why AuroRA adds nonlinearity

Because \(A\) and \(B\) are linear layers:

\[
B(Ax)=(BA)x
\]

Therefore, the LoRA adapter is ultimately a low-rank linear correction.

AuroRA uses:

\[
B(\operatorname{GELU}(Ax))
\]

Because GELU is nonlinear, the two matrices cannot be combined into one linear matrix.

GELU changes the intermediate values depending on their magnitude and sign. This allows AuroRA to:

- suppress some features;
- preserve other features;
- create input-dependent corrections;
- model more complex interactions between features.

For sentiment classification, this may help with context-dependent patterns such as:

```text
good
```

versus:

```text
not good
```

The same word can contribute differently depending on the surrounding context.

AuroRA is therefore intended to provide a more expressive adapter while still training far fewer parameters than full fine-tuning.

---

## 13. Side-by-side comparison

| Feature | Full fine-tuning | LoRA | AuroRA |
|---|---|---|---|
| Original RoBERTa weights | Updated | Frozen | Frozen |
| Uses low-rank matrices | No | Yes | Yes |
| Temporary bottleneck | No | \(768\rightarrow8\rightarrow768\) | \(768\rightarrow8\rightarrow768\) |
| Nonlinear activation in adapter | No | No | Yes |
| Query/value adapters | Not required | Yes | Planned |
| Classifier trained | Yes | Yes | Planned |
| Trainable parameter count | Approximately 125M | 887,042 | Similar small count |
| Main advantage | Maximum flexibility | Efficiency | More expressive efficient adaptation |

---

## 14. One-sentence presentation explanation

> Full fine-tuning updates the entire RoBERTa model, LoRA freezes RoBERTa and learns a low-rank linear correction using \(A\) and \(B\), while AuroRA keeps the same efficient low-rank structure but adds GELU between \(A\) and \(B\) to create a nonlinear correction.

---

## 15. Important limitation

AuroRA is not guaranteed to achieve higher accuracy.

Its purpose is to test whether adding nonlinear behavior can improve the accuracy-efficiency tradeoff while keeping the number of trainable parameters small.

The final conclusion must be based on the measured:

- accuracy;
- F1-score;
- training time;
- trainable parameters;
- memory usage.