from enum import Enum

class PeftType(Enum):
    BASE = "base"
    LORA = "lora"
    TANH_AURORA = "tanh-aurora"
    TRAINABLE_TANH_AURORA = "trainable-tanh-aurora"