# LoRA Setup and Debugging Writeup

## 1. Purpose

Before running the complete LoRA experiment, a smaller debug experiment was performed.

The purpose was to verify that:

- the LoRA model loaded correctly;
- LoRA adapters were added to RoBERTa;
- original RoBERTa parameters were frozen;
- the sentiment classifier was trainable;
- the training pipeline executed correctly;
- validation and F1-score calculations worked;
- checkpoints could be saved.

The debug experiment used only a small subset of SST-2 so that problems could be identified quickly.

---

## 2. LoRA Configuration

The base model was:

```text
roberta-base
```

The LoRA configuration was:

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
    bias="none",
    task_type="SEQ_CLS"
)
```

### Meaning of the configuration

- `r=8`: LoRA rank. This controls the size and flexibility of the LoRA update.
- `lora_alpha=16`: Scaling factor for the LoRA update.
- `lora_dropout=0.1`: Randomly drops 10% of values in the LoRA path during training.
- `target_modules=["query", "value"]`: Adds LoRA to the query and value projections in the attention layers.
- `modules_to_save=["classifier"]`: Keeps the sentiment classifier trainable and saves it with the LoRA model.
- `bias="none"`: Does not train bias parameters.
- `task_type="SEQ_CLS"`: Specifies sequence classification.

---

## 3. LoRA Model-Creation Code

The LoRA model was created using:

```python
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


def create_lora_model(device):
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
        bias="none",
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)
    model = model.to(device)

    model.print_trainable_parameters()

    return model
```

The following line applied the LoRA configuration:

```python
model = get_peft_model(model, lora_config)
```

This froze the original RoBERTa parameters and added LoRA adapters.

The following line moved the model to Apple MPS or CPU:

```python
model = model.to(device)
```

The following line displayed the parameter counts:

```python
model.print_trainable_parameters()
```

---

## 4. Parameter Check

The output was:

```text
trainable params: 887,042
all params: 125,534,212
trainable%: 0.7066
```

Only 887,042 parameters were trainable, while the complete LoRA-wrapped model contained 125,534,212 parameters.

The trainable percentage was:

\[
\frac{887{,}042}{125{,}534{,}212}
\times 100
=
0.7066\%
\]

Therefore, more than 99% of the model parameters remained frozen.

---

## 5. Trainable-Parameter Breakdown

RoBERTa has 12 Transformer layers. Each layer contains a query projection and a value projection.

For one targeted module:

\[
A:8 \times 768
\]

\[
B:768 \times 8
\]

Therefore, one LoRA module contains:

\[
(8 \times 768)+(768 \times 8)
=
12{,}288
\]

parameters.

There are 12 layers and 2 target modules per layer:

\[
12 \times 2 \times 12{,}288
=
294{,}912
\]

LoRA parameters.

The trainable sentiment classifier contains:

\[
768 \times 768 + 768
\]

parameters in the dense layer, and:

\[
768 \times 2 + 2
\]

parameters in the output layer.

Therefore, the classifier contains:

\[
592{,}130
\]

parameters.

The total trainable parameter count is:

\[
294{,}912+592{,}130
=
887{,}042
\]

This matches the number printed by PEFT.

---

## 6. Debug Configuration

The debug configuration was:

```python
DEBUG = True
TRAIN_MODEL = True
EPOCHS = 1
LEARNING_RATE = 2e-5
```

The debug subset contained:

- 128 training examples;
- 128 validation examples;
- batch size of 8;
- 16 training batches;
- 16 validation batches.

The purpose of this configuration was to test the pipeline quickly instead of waiting for the complete SST-2 run.

---

## 7. First Debug Run

The first debug run used one epoch and a learning rate of `2e-5`.

Results:

```text
trainable params: 887,042
all params: 125,534,212
trainable%: 0.7066

Validation loss before training: 0.70689981803298
Validation accuracy before training: 0.4765625
Validation F1-score before training: 0.0

Epoch 1/1 training loss: 0.6945815868675709

Training time: 3.277046459 seconds
Additional MPS memory: 13.188232421875 MB

Validation loss: 0.69985106959939
Validation accuracy: 0.4765625
Validation F1-score: 0.0
```

The loss decreased slightly, but accuracy and F1-score did not improve.

---

## 8. Two-Epoch Debug Run

The model was then trained for two epochs using learning rate `2e-5`.

Results:

```text
Epoch 1/2 training loss: 0.6945815868675709
Epoch 2/2 training loss: 0.6992736421525478

Validation loss: 0.6980736665427685
Validation accuracy: 0.4765625
Validation F1-score: 0.0
```

The second epoch did not improve the training loss. Accuracy and F1-score remained unchanged.

---

## 9. Learning-Rate Debug Run

A learning rate of `1e-4` was tested for five epochs.

Results:

```text
Epoch 1/5 training loss: 0.6991285495460033
Epoch 2/5 training loss: 0.6974690780043602
Epoch 3/5 training loss: 0.6926314570009708
Epoch 4/5 training loss: 0.6644188985228539
Epoch 5/5 training loss: 0.7078471891582012

Validation loss: 0.6904580928385258
Validation accuracy: 0.4765625
Validation F1-score: 0.0
```

The model showed temporary improvement during the fourth epoch, but the training loss increased during the fifth epoch.

This suggested that the small debug subset was not sufficient for judging final model performance.

---

## 10. Prediction-Count Diagnostic

Because the validation accuracy and F1-score did not change, a temporary diagnostic was added to count predictions from each class.

Before the validation loop:

```python
prediction_counts = torch.zeros(2, dtype=torch.long)
```

After obtaining predictions:

```python
predictions = outputs.logits.argmax(dim=1)

prediction_counts += torch.bincount(
    predictions.cpu(),
    minlength=2
)
```

After validation:

```python
print("Predicted class counts:", prediction_counts.tolist())
```

The output was:

```text
Predicted class counts: [128, 0]
```

This means that the model predicted class 0 for all 128 validation examples.

This explained the metrics:

- Accuracy was `0.4765625`, which corresponded to the proportion of class 0 examples.
- F1-score was `0.0` because the model never predicted class 1.

---

## 11. Classifier-Trainability Diagnostic

To verify that the sentiment classifier was trainable, the following temporary code was added:

```python
for name, parameter in model.named_parameters():
    if parameter.requires_grad and "classifier" in name:
        print("Trainable:", name, parameter.shape)
```

The output included:

```text
Trainable: base_model.model.classifier.modules_to_save.default.dense.weight
Trainable: base_model.model.classifier.modules_to_save.default.dense.bias
Trainable: base_model.model.classifier.modules_to_save.default.out_proj.weight
Trainable: base_model.model.classifier.modules_to_save.default.out_proj.bias
```

This confirmed that the classifier was trainable.

Therefore, the debugging issue was not caused by the classifier being frozen.

---

## 12. Debugging Conclusion

The debug experiments confirmed that:

- the LoRA configuration was valid;
- LoRA adapters were successfully added;
- original RoBERTa weights were frozen;
- the classifier was trainable;
- the optimizer and training loop executed;
- validation metrics were calculated correctly;
- the checkpoint pipeline worked.

The small debug experiments produced poor classification results because the debug subset was too small to reliably evaluate model quality.

The model predicted only one class during validation, even though its loss changed slightly.

The debug results were used only to verify implementation. They were not used as the final LoRA performance result.

---

## 13. Temporary Debugging Code Removed

After the investigation, the following temporary code was removed:

```python
prediction_counts = torch.zeros(2, dtype=torch.long)
```

```python
prediction_counts += torch.bincount(
    predictions.cpu(),
    minlength=2
)
```

```python
print("Predicted class counts:", prediction_counts.tolist())
```

The classifier-trainability diagnostic was also removed:

```python
for name, parameter in model.named_parameters():
    if parameter.requires_grad and "classifier" in name:
        print("Trainable:", name, parameter.shape)
```

The following permanent code was kept:

```python
model.print_trainable_parameters()
```

and:

```python
modules_to_save=["classifier"]
```

---

## 14. Full LoRA Run

After the debug checks were complete, the model was trained using the complete SST-2 training set.

The final LoRA run used:

```text
Learning rate: 2e-5
Epochs: 1
Training examples: 67,349
Validation examples: 872
```

Results:

```text
Trainable parameters: 887,042
Total parameters: 125,534,212
Trainable percentage: 0.7066%

Training loss: 0.2958758710074361
Training time: 835.1681627500001 seconds
Additional MPS memory: 13.196044921875 MB

Validation loss: 0.19965771757442197
Validation accuracy: 0.9254587155963303
Validation F1-score: 0.9276974416017798
```

---

## 15. Final Interpretation

The debug experiments were useful for verifying the implementation, but they were not suitable for measuring final model quality.

The full-data LoRA experiment showed that:

- LoRA successfully adapted RoBERTa for sentiment classification;
- LoRA achieved more than 92% validation accuracy;
- only 0.7066% of model parameters were trainable;
- the complete dataset was necessary for learning a reliable decision boundary;
- the poor debug result did not indicate a failure of the LoRA implementation.

The final LoRA result was:

- Validation accuracy: 92.55%;
- Validation F1-score: 92.77%;
- Training time: 835.17 seconds;
- Trainable parameters: 887,042;
- Trainable percentage: 0.7066%.