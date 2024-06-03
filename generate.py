import argparse
import time
import random as r

import jax
from sentencepiece import SentencePieceProcessor
from model.model import Transformer, load_object, build_model
from model.sample import GenerateTokens
import keras as nn
nn.mixed_precision.set_dtype_policy("mixed_bfloat16")

parser = argparse.ArgumentParser()
parser.add_argument('--num_times', default="1")
parser.add_argument('--model', default="15M")
parser.add_argument('--prompt', default="Once upon a time")
args = parser.parse_args()

# currently supports only 15M model
model_name = args.model # 280K, 15M, 45M, 110M
tokenizer_path = "sentence_piece_32000.model"
if model_name == "280K":
    tokenizer_path = "sentence_piece_512.model"
    model = ...
    train_state = ...
elif model_name == "15M":
    import config.config_15M as config
    model = build_model(
        Transformer(causal=True, config=config.GPTConfig()), 
        (2, config.GPTConfig.maxlen), 
        (0, config.GPTConfig.vocab_size-1)
    )
    train_state = load_object(path="ckpt/stories32000/train_state_15M.gpt")
elif model_name == "45M":
    model = ...
    train_state = ...
elif model_name == "110M":
    model = ...
    train_state = ...
else:
    raise ValueError("280K, 15M, 45M, 110M are the only possible ")

spm = SentencePieceProcessor(model_file=tokenizer_path)
SOS = spm.bos_id()
sampler = GenerateTokens(model, train_state, config.GPTConfig.maxlen, sos=SOS)

start_toks = jax.numpy.array([[SOS]+spm.Encode(str(args.prompt))])
assert len(start_toks) < config.GPTConfig.maxlen

for _ in range(int(args.num_times)):
    seed = r.randint(0, 10000)
    print("Random Seed:", seed)
    t0 = time.time()
    out_toks = sampler.generate(
        idx=start_toks,
        top_p=0.9,
        seed=seed
    )
    print(spm.Decode(out_toks))
    t1 = time.time()
    sec = t1-t0
    print("\nTime Taken (sec):", sec)
    print("Tokens per second:", len(out_toks)/sec, "\n\n")