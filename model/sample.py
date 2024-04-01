
import jax
import keras as nn
from jax import numpy as jnp, Array, random as jrand
from typing import Sequence
from functools import partial

class GenerateTokens:
    def __init__(self, model:nn.Model, train_state:Sequence[list], maxlen:int, sos:int):
        self.SOS = jnp.array([[sos]])
        self.model = model
        self.train_state = train_state
        self.maxlen = maxlen
    
    @staticmethod
    @jax.jit
    def randCategorical(key:Array, logits:Array):
        idx_next = jrand.categorical(key, logits)[..., None]
        return idx_next # (1, 1)

    @staticmethod
    # TODO(VachanVY): Debug when jitted; does it speed up code?
    # @partial(jax.jit, static_argnums=-1)
    def topK(key:Array, logits:Array, k:Array):
        if jnp.where(k==1, True, False):
            idx_next = logits.argmax(-1, keepdims=True)
            return idx_next
        logits, topK_idx = jax.lax.top_k(logits, k=k)
        idx = jrand.categorical(key, logits)
        idx_next = topK_idx[0][idx][..., None]
        return idx_next # (1, 1)

    @staticmethod
    # TODO(VachanVY): Debug when jitted; does it speed up code?
    # @partial(jax.jit, static_argnums=(-1, -2))
    def topP(key:Array, logits:Array, p:float, k:int|None=None):
        probs = jax.nn.softmax(logits, axis=-1)
        # reverse arg sort of probs
        rev_argsort_prob_idx = jnp.argsort(probs)[:, ::-1]
        # True bools of idx less than p that sum less than p || False bool of least idxs that sum more than p
        less_than_p_bools = (jnp.cumsum(probs.take(rev_argsort_prob_idx), axis=-1) <= p)

        # idx from which to mask
        mask_from_id = less_than_p_bools.sum()+1
        # idxs to mask
        mask_idx = rev_argsort_prob_idx[:, mask_from_id:]
        # -inf masked idx won't be sampled 
        logits = logits.at[:, mask_idx].set(-jnp.inf)

        if k is not None:
            key, _ = jrand.split(key)
            idx_next = GenerateTokens.topK(key, logits, k)
            return idx_next # (1, 1)
        
        idx_next = jrand.categorical(key, logits)[..., None]
        return idx_next # (1, 1)

    @partial(jax.jit, static_argnums=0)
    def get_logits(self, idx_cond:Array, trainable_variables:list, non_trainable_variables:list):
        logits, non_trainable_variables = self.model.stateless_call(
            trainable_variables, non_trainable_variables, idx_cond
        )
        return logits[:, -1, :], non_trainable_variables

    def generate(self, idx:Array, max_tok:int=500, top_k:int|None=None, top_p:float|None=None,  temp:float=0.8, seed:int=42):
        trainable_variables, non_trainable_variables = self.train_state
        if (not top_k) and (not top_p):
            sampleTok = GenerateTokens.randCategorical
        elif (top_k is not None) and (top_p is not None):
            sampleTok = lambda key, logits: GenerateTokens.topP(key, logits, top_p, top_k)
        elif top_k is not None:
            sampleTok = lambda key, logits: GenerateTokens.topK(key, logits, top_k)
        elif top_p is not None:
            sampleTok = lambda key, logits: GenerateTokens.topP(key, logits, top_p)
        else:
            assert False, "(?_?) ¯\(°_o)/¯"
        
        key = jrand.PRNGKey(seed)
        for _ in range(max_tok):
            idx_cond = idx[:, -self.maxlen:] # (B=1, T)
            logits, non_trainable_variables = self.get_logits(
                idx_cond, trainable_variables, non_trainable_variables
            ) # (B, vocab_size)
            idx_next = sampleTok(key, logits/temp) # (1, 1)
            if idx_next == self.SOS:
                break
            key, _ = jrand.split(key)
            idx = jnp.concatenate((idx, idx_next), axis=-1) # (B=1, T+1)
        return idx[0].tolist()