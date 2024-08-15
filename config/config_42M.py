from dataclasses import dataclass
from typing import Callable
import keras as nn

@dataclass
class GPTConfig:
    """GPT 15M Configuration"""
    use_flash_att:bool=True
    d_model:int = 512
    num_layers:int = 8
    num_heads:int = 8
    maxlen:int = 256
    vocab_size:int = 32_000
    output_units:int = None
    assert d_model % 2 == 0
    assert d_model % num_heads == 0
    dropout_rate:float = 0.0
    use_bias:bool = True
    intializer:Callable = lambda std: nn.initializers.RandomNormal(mean=0.0, stddev=std)

