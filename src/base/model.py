from transformers import AutoModelForSequenceClassification
from transformers.utils import logging


def create_model():
    logging.set_verbosity_error()

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2
    )

    return model