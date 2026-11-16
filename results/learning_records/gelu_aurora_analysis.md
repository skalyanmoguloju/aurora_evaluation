# GELU-Based AuroRA Analysis

## 1. Purpose

After completing the full fine-tuning and LoRA baselines, we implemented a simplified AuroRA adapter by adding a GELU activation between the two low-rank projection matrices.

This experiment was designed to test whether adding a nonlinear mapping could improve the accuracy-efficiency tradeoff of LoRA.

This implementation is a simplified nonlinear LoRA variant. It is not the complete Adaptive Nonlinear Layer implementation from the AuroRA paper.

---

## 2. Adapter Structure

The LoRA update is:

\[
\Delta h_{\text{LoRA}}=sB(Ax)
\]

The GELU-based AuroRA update is:

\[
\Delta h_{\text{GELU-AuroRA}}
=
sB(\operatorname{GELU}(Ax))
\]

The complete layer output is:

\[
h
=
W_0x
+
sB(\operatorname{GELU}(Ax))
\]

where:

- \(W_0x\) is the original frozen RoBERTa transformation;
- \(A\) projects the input into a lower-dimensional space;
- GELU applies a nonlinear transformation;
- \(B\) projects the result back to the original output dimension;
- \(s\) controls the strength of the adapter update.

For RoBERTa-base:

\[
768\rightarrow8\rightarrow8\rightarrow768
\]

The first and last dimensions represent the RoBERTa hidden size. The middle dimension is the adapter rank.

---

## 3. Model Integration

The GELU-based AuroRA adapter was added to the same modules used by LoRA:

- query projections in all 12 Transformer layers;
- value projections in all 12 Transformer layers.

Therefore, the model contained:

\[
12\times2=24
\]

AuroRA-wrapped modules.

The original RoBERTa encoder weights were frozen. The trainable parameters were:

- AuroRA down-projection matrices;
- AuroRA up-projection matrices;
- the sentiment classification head.

The model continued to use the same SST-2 tokenization, DataLoaders, validation procedure, and classification head as the other experiments.

---

## 4. Parameter Results

The GELU-based AuroRA model contained:

```text
Trainable parameters: 887,042
Total parameters: 124,942,082
Trainable percentage: 0.7099625568909601%
```

The trainable parameter count was the same as LoRA because both methods used:

- rank 8;
- query and value adapters;
- a trainable classification head.

The total parameter count differed slightly because the custom AuroRA implementation did not create the duplicated classifier structure used by PEFT.

---

## 5. Full GELU-Based AuroRA Result

The experiment used:

```text
Model: roberta-base
Dataset: GLUE SST-2
Learning rate: 2e-5
Epochs: 1
Batch size: 8
Maximum sequence length: 128
Rank: 8
Target modules: query and value
Activation: GELU
```

Results:

```text
Training loss: 0.3099292336258126
Training time: 812.065555709 seconds
Additional MPS memory: 15.524169921875 MB

Validation loss: 0.20352300628165312
Validation accuracy: 0.9243119266055045
Validation F1-score: 0.9266666666666666
```

Therefore:

- Validation accuracy: 92.43%
- Validation F1-score: 92.67%
- Training time: 812.07 seconds
- Additional MPS memory: 15.52 MB
- Trainable parameters: 887,042
- Trainable percentage: 0.7100%

---

## 6. Comparison with Other Methods

| Metric | Full Fine-Tuning | LoRA | GELU-Based AuroRA |
|---|---:|---:|---:|
| Validation accuracy | 92.78% | 92.55% | 92.43% |
| Validation F1-score | 93.05% | 92.77% | 92.67% |
| Training time | 1,400.53 sec | 835.17 sec | 812.07 sec |
| Trainable parameters | Approximately 125M | 887,042 | 887,042 |
| Trainable percentage | Approximately 100% | 0.7066% | 0.7100% |
| Additional MPS memory | 1,736.57 MB | 13.20 MB | 15.52 MB |

---

## 7. Accuracy Finding

The GELU-based AuroRA model achieved:

- 92.43% validation accuracy;
- 92.67% validation F1-score.

Compared with LoRA:

- accuracy decreased by approximately 0.12 percentage points;
- F1-score decreased by approximately 0.10 percentage points.

Therefore, adding GELU did not improve predictive performance in this experiment.

The result suggests that the nonlinear activation did not provide an advantage under the current configuration:

- one training epoch;
- rank 8;
- learning rate \(2e-5\);
- query/value adaptation;
- one random seed;
- SST-2 only.

This does not prove that nonlinear adapters are ineffective in general. It only shows that the simplified GELU-based implementation did not outperform LoRA under these experimental conditions.

---

## 8. Efficiency Finding

The GELU-based AuroRA model trained in:

\[
812.07\text{ seconds}
\]

LoRA trained in:

\[
835.17\text{ seconds}
\]

Therefore, GELU-based AuroRA was approximately 23 seconds faster in this run.

However, this difference is small and may be affected by system timing variation. It should not be presented as a strong efficiency advantage.

Compared with full fine-tuning, GELU-based AuroRA was approximately 42% faster and trained only about 0.71% of the model parameters.

---

## 9. Memory Finding

The measured additional MPS memory was:

- LoRA: 13.20 MB;
- GELU-based AuroRA: 15.52 MB.

GELU-based AuroRA used slightly more additional memory than LoRA. This is expected because AuroRA adds a nonlinear operation and an additional intermediate computation.

The memory difference was small, but the measurements should be interpreted carefully because MPS memory allocation can vary between runs.

---

## 10. Main Finding

The simplified GELU-based AuroRA model produced performance very close to LoRA:

\[
92.43\%\text{ accuracy}
\quad\text{versus}\quad
92.55\%\text{ for LoRA}
\]

However, it did not outperform LoRA.

The main conclusion is:

> The GELU-based AuroRA adapter successfully integrated nonlinear processing into the LoRA-style update and maintained comparable sentiment-classification performance, but it did not provide an accuracy improvement over standard LoRA on SST-2.

---

## 11. Important Limitation

This experiment should not be described as a complete reproduction of the AuroRA paper.

The current implementation uses:

\[
B(\operatorname{GELU}(Ax))
\]

The paper’s full AuroRA method uses an Adaptive Nonlinear Layer containing:

- a fixed nonlinear component based on tanh;
- a learnable B-spline component;
- a compressed hidden dimension smaller than the standard LoRA rank.

Therefore, the current experiment should be called:

```text
GELU-based AuroRA
```

or:

```text
simplified nonlinear LoRA
```

A separate paper-faithful implementation is required before making claims about the complete AuroRA method.

---

## 12. Final Conclusion

The GELU-based AuroRA implementation was successful from an engineering perspective:

- the custom adapter was created;
- the adapter preserved tensor shapes;
- the wrapper worked inside RoBERTa;
- all 24 query/value modules were replaced;
- the model trained successfully;
- the classifier produced valid predictions.

From a performance perspective:

- full fine-tuning achieved the highest accuracy;
- LoRA slightly outperformed the GELU-based AuroRA variant;
- GELU-based AuroRA remained highly parameter-efficient;
- nonlinear processing alone did not guarantee better performance.

The next experiment should implement the complete Adaptive Nonlinear Layer from the AuroRA paper.