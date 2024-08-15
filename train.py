from dataclasses import dataclass
from functools import partial
import functools
from typing import Optional, Sequence, Any
import glob
import random as r
import os
import operator
import numpy as np
from tqdm import tqdm
import time
import pickle
import matplotlib.pyplot as plt

import torch
import jax
from jax import (
    Array,
    numpy as jnp,
)
try:
    from flash_attn_jax import flash_mha; del flash_mha
    USE_FLASH_ATT = True
except:
    USE_FLASH_ATT = False
import keras as nn; nn.utils.set_random_seed(42)
nn.mixed_precision.set_dtype_policy("mixed_bfloat16")

from sentencepiece import SentencePieceProcessor
from model.model import Transformer, build_model, load_object
from config.config_42M import GPTConfig



DATA_CACHE_DIR = "data/TinyStories"
TRAIN_FILE_PATH = os.path.join(DATA_CACHE_DIR, "train.txt")
VAL_FILE_PATH = os.path.join(DATA_CACHE_DIR, "val.txt")

SHARD_DIR = os.path.join(DATA_CACHE_DIR, f"tok{GPTConfig.vocab_size}")

def save_object(dir_suffix_ftype:str, obj:Any):
    """
    dir_suffix_ftype: directory suffix and file type separated by "|"
    obj: Anything which is to be stored
    """
    dir, suffix, ftype = dir_suffix_ftype.split("|"); path = os.path.join(dir, "".join([suffix, f".{ftype}"]))
    os.makedirs(name=dir, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj=obj, file=file, protocol=pickle.HIGHEST_PROTOCOL)
    return path


@dataclass
class TArgs:
    batch_size:int = 32 # micro-mini-batch_size if num_grad_accumalation_steps > 1
    num_grad_accumalation_steps:int = 16
    ## num_tok_per_update = batch_size * maxlen * gradient_accumalation = 32 * 256 * 16 = 131_072

    # lr scheduler
    init_lr:float = 1e-7
    max_lr:float = 5e-4
    min_lr:float = 0.0*max_lr # The factor is usually 0.1 or 0.0
    num_steps:int = 100_000
    warmup_steps:int = 1000
    decay_steps:int = num_steps

    # optimizer
    beta1:float = 0.9
    beta2:float = 0.95
    weight_decay:float = 1e-1
    clipnorm:float = 1e0

    # training
    resume_from_checkpoint:Optional[str] = "ckpt/stories32000/checkpoint42M.gpt"
    return_best_train_states:bool = True
    checkpoint_dir:str = "ckpt/stories32000"
    eval_freq:int = 2000
    eval_steps:int = 100
    patience:int = 15 # early stopping with restore best weights


spm = SentencePieceProcessor(model_file="sentence_piece_32000.model")
SOS = spm.bos_id()

def pretokenize_and_save_dataset(train_ds_path:str, val_ds_path:str, num_shards:int):
    if glob.glob(os.path.join(SHARD_DIR, "*.npy")):
        print("Dataset is already pretokenized.")
    else:
        print("Pretokenizing dataset...")
        dataset = open(train_ds_path, "r", encoding="utf-8").read().split("<|endoftext|>")
        val_dataset = open(val_ds_path, "r", encoding="utf-8").read().split("<|endoftext|>")

        dataset = dataset + val_dataset; del val_dataset
        dataset = list(map(str.strip, dataset))
        dataset:list = spm.Encode(
                dataset,
                add_bos=True,
                add_eos=False
            ) # [[SOS story], ..., [SOS story]]
        print("Dataset:")
        print("\tNumber of stories:", len(dataset))

        # flatten
        dataset = functools.reduce(operator.iconcat, dataset, [])
        num_tokens = len(dataset); print("\tNumber of tokens in the dataset:", num_tokens)
        print("\tNumber of unique tokens in the dataset:", len(set(dataset)))
        
        dataset = np.asarray(dataset, dtype=np.uint16) # [SOS story ... SOS story]
        print("\tAvg length of story:", num_tokens/((dataset==SOS).sum()))

        # shard and save dataset
        sharded_datasets_list = np.array_split(dataset, num_shards) # [[SOS story...], [...], [...], ...]
        filenames = [os.path.join(SHARD_DIR, f"shard{i+1}.npy") for i in range(num_shards)]
        
        for filename, sharded_ds in tqdm(zip(filenames, sharded_datasets_list), total=len(filenames), desc="Saving pretokenized shards"):
            with open(filename, "wb") as f:
                np.save(f, sharded_ds)
        print("Done.")


class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, split:str, maxlen:int, seed:int=42):
        self.split = split
        self.maxlen = maxlen
        
        os.makedirs(SHARD_DIR, exist_ok=True)
        self.shard_filepaths = glob.glob(os.path.join(SHARD_DIR, "*.npy"))
        self.r = r.Random(seed)

    def __iter__(self):
        split_shard_filepaths = self.shard_filepaths[:-1] if self.split == "train" else self.shard_filepaths

        while True:
            self.r.shuffle(split_shard_filepaths)
            for shard in split_shard_filepaths:
                m:np.ndarray = np.load(shard, mmap_mode="r")

                num_batches = len(m)//self.maxlen
                num_batches -= 1 # drop remainder
                assert num_batches > 0, "Number of batches should be greater than 0. Investigate..."

                ixs = list(range(num_batches))
                self.r.shuffle(ixs)

                for ix in ixs:
                    start = ix*self.maxlen
                    end = start + self.maxlen + 1

                    chunk = torch.from_numpy(m[start:end].astype(dtype=np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


class BatchedDataset:
    @staticmethod
    def iter_ds(batch_size, device, num_workers=0, **ds_kwargs):
        ds = torch.utils.data.DataLoader(
            IterDataset(**ds_kwargs), batch_size=batch_size, pin_memory=True,
            num_workers=num_workers
        )

        for x, y in ds:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            yield x, y


def main():
    pretokenize_and_save_dataset(TRAIN_FILE_PATH, VAL_FILE_PATH, num_shards=50)

    ds_iterater = partial(
        BatchedDataset.iter_ds,
        batch_size=TArgs.batch_size,
        device="cpu",
        num_workers=0,
        maxlen=GPTConfig.maxlen,
        seed=42
    )
    train_iterator, val_iterator = (
        ds_iterater(split="train"),
        ds_iterater(split="val")
    )    
    
    model = Transformer(causal=True, config=GPTConfig(use_flash_att=USE_FLASH_ATT), output_activation=None)
    model = build_model(model, (2, GPTConfig.maxlen), (0, GPTConfig.vocab_size-1)); print("\n\n")
    model.summary(); print("\n\n")

    class LearningRateSchedule:
        def __init__(self, start_iter:int):
            self.start_iter = start_iter
            self.learning_rate = nn.optimizers.schedules.CosineDecay(
                initial_learning_rate=TArgs.min_lr,
                decay_steps=TArgs.decay_steps,
                warmup_steps=TArgs.warmup_steps,
                warmup_target=TArgs.max_lr,
                alpha=TArgs.min_lr/TArgs.max_lr
            )

        def __call__(self, step:int):
            return self.learning_rate(step+self.start_iter)
        

    class ParamGradManager:
        """Filter and Combine Gradients and Parameters for decay and no-decay variables"""
        def __init__(self, trainable_vars:list):
            order_before_segregate = [v.path for v in trainable_vars]
            order_after_segregate = (
                [v.path for v in trainable_vars if len(v.shape)!=1] +
                [v.path for v in trainable_vars if len(v.shape)==1]
            )
            self.idx = [order_after_segregate.index(b) for b in order_before_segregate]

        def filter_obj(self, trainable_obj:list):
            """can be grads or params"""
            decay_obj = [v for v in trainable_obj if len(v.shape)!=1]
            nodecay_obj = [v for v in trainable_obj if len(v.shape)==1]
            return decay_obj, nodecay_obj
        
        def combine_obj(self, decay_obj:list, nodecay_obj:list):
            obj = decay_obj + nodecay_obj
            return [obj[i] for i in self.idx]
        
    start_iter = 0
    training_losses = {"train": []}
    if TArgs.resume_from_checkpoint is not None:
        print("Resuming from checkpoint at:", TArgs.resume_from_checkpoint, "...")
        best_ckpt = load_object(TArgs.resume_from_checkpoint)
        (
            trainable_vars,
            non_trainable_vars,
            opt_vars,
            start_iter,
            best_val_loss,
            # training_losses
        ) = best_ckpt
        (decay_opt_vars, nodecay_opt_vars) = opt_vars
        for v, a in zip(model.trainable_variables, trainable_vars):
            v.assign(a)
        trainable_vars = model.trainable_variables
        param_grad_manager = ParamGradManager(trainable_vars)


    learning_rate = LearningRateSchedule(start_iter=start_iter)
    adamw = lambda weight_decay: nn.optimizers.AdamW(
            learning_rate=learning_rate,
            beta_1=TArgs.beta1,
            beta_2=TArgs.beta2,
            clipnorm=TArgs.clipnorm,
            weight_decay=weight_decay
    )
    decay_optimizer = adamw(weight_decay=TArgs.weight_decay)
    nodecay_optimizer = adamw(weight_decay=0.0)
    loss_fn = nn.losses.SparseCategoricalCrossentropy(from_logits=True)

    step = 0; wait = 0
    if TArgs.resume_from_checkpoint is None:
        trainable_vars = model.trainable_variables
        non_trainable_vars = model.non_trainable_variables
        
        param_grad_manager = ParamGradManager(trainable_vars)
        
        decay_opt_vars, nodecay_opt_vars = decay_optimizer.variables, nodecay_optimizer.variables
        best_val_loss = 1e8
        
        best_ckpt = (
            trainable_vars,
            non_trainable_vars,
            (decay_opt_vars, nodecay_opt_vars),
            step,
            best_val_loss,
            # training_losses
        )

    for param, opt in zip(param_grad_manager.filter_obj(trainable_vars), [decay_optimizer, nodecay_optimizer]):
        opt.build(param)

    @jax.jit
    def get_accuracy(y_true:Array, logits:Array): # (B, T), (B, T, vocab_size)
        batched_num_correct = (logits.argmax(-1)==y_true).sum(-1)/y_true.shape[-1] # (B,)
        accuracy = batched_num_correct.mean()
        return accuracy

    @jax.jit
    def compute_loss(
            trainable_vars:list,
            non_trainable_vars:list,
            X_batch:Array, y_batch:Array,
            num_grad_accumalation_steps:int
        ):
        logits, non_trainable_vars = model.stateless_call(
            trainable_vars,  non_trainable_vars,
            X_batch)
        loss = loss_fn(y_batch, logits)
        accuracy = get_accuracy(y_batch, logits)
        unscaled_loss = loss/num_grad_accumalation_steps
        # scaled_loss = optimizer.scale_loss(unscaled_loss)
        return unscaled_loss, (unscaled_loss, accuracy, non_trainable_vars)
    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

    @partial(jax.jit, static_argnums=-1)
    def mini_step(train_state:Sequence[list], X_batch:Array, y_batch:Array, num_grad_accumalation_steps:int):
        trainable_vars, non_trainable_vars = train_state

        (_, aux), grad = grad_fn(
            trainable_vars, non_trainable_vars, X_batch, y_batch,
            num_grad_accumalation_steps
        )
        (unscaled_loss, accuracy, non_trainable_vars) = aux
        return grad, (unscaled_loss, accuracy), (trainable_vars, non_trainable_vars)

    def optimizer_apply(optimizer, opt_vars, grads, trainable_vars):
        trainable_vars, opt_vars = optimizer.stateless_apply(opt_vars, grads, trainable_vars)
        return trainable_vars, opt_vars

    decay_opt_apply = jax.jit(fun=lambda opt_vars, grads, trainable_vars: optimizer_apply(
        decay_optimizer, opt_vars, grads, trainable_vars
    ))
    nodecay_opt_apply = jax.jit(fun=lambda opt_vars, grads, trainable_vars: optimizer_apply(
        nodecay_optimizer, opt_vars, grads, trainable_vars
    ))

    def update_params(
        grads:list,
        trainable_vars:list,
        optimizer_vars:tuple[list, list],
    ):
        decay_grads, nodecay_grads = param_grad_manager.filter_obj(grads)
        decay_trainable_vars, nodecay_trainable_vars = param_grad_manager.filter_obj(trainable_vars)
        decay_opt_vars, nodecay_opt_vars = optimizer_vars
        
        decay_trainable_vars, decay_opt_vars = decay_opt_apply(
            decay_opt_vars, decay_grads, decay_trainable_vars
        )
        nodecay_trainable_vars, nodecay_opt_vars = nodecay_opt_apply(
            nodecay_opt_vars, nodecay_grads, nodecay_trainable_vars
        )
        trainable_vars1 = param_grad_manager.combine_obj(decay_trainable_vars, nodecay_trainable_vars)
        assert (
            [v.shape for v in trainable_vars1] == 
            [v.shape for v in trainable_vars]), (
                f"train vars aft: {[v.shape for v in trainable_vars1]}\n\ntrain vars bef: {[v.shape for v in trainable_vars]}"
                )
        return trainable_vars1, (decay_opt_vars, nodecay_opt_vars)

    def evaluate(train_state:Sequence[list]):
        trainable_vars, non_trainable_vars = train_state
        mean_losses = []; mean_accuracies = []
        for eval_batch_iter in [train_iterator, val_iterator]:
            X_batch, y_batch = next(eval_batch_iter)
            losses = jnp.empty(TArgs.eval_steps)
            accuracies = jnp.empty_like(losses)

            for eval_step in range(TArgs.eval_steps):
                _, (unscaled_loss, accuracy, non_trainable_vars) = compute_loss(
                    trainable_vars, non_trainable_vars,
                    jnp.array(X_batch), jnp.array(y_batch), 1
                )
                losses = losses.at[eval_step].set(unscaled_loss)
                accuracies = accuracies.at[eval_step].set(accuracy)
                X_batch, y_batch = next(eval_batch_iter)
            mean_losses.append(losses.mean())
            mean_accuracies.append(accuracies.mean())
        return mean_losses, mean_accuracies # ([train_loss, val_loss], [train_accuracy, val_accuracy])
    

    wait = 0
    best_step = step

    t0 = time.time()
    print("Training about to start...")
    X_batch, y_batch = next(train_iterator)
    # TODO: Optimize Train Loop to reduce time per step
    while True:
        # condition to terminate
        if step > (TArgs.num_steps-start_iter) or wait > TArgs.patience:
            print(f"Early Stopping at Step {step}." if wait > TArgs.patience else "Training Terminated.")
            break
        
        # train model
        grads = jax.tree_util.tree_map(jnp.zeros_like, trainable_vars)
        for _ in range(TArgs.num_grad_accumalation_steps):
            grad, (loss, accuracy), (trainable_vars, non_trainable_vars) = mini_step(
                (trainable_vars, non_trainable_vars),
                jnp.array(X_batch), jnp.array(y_batch),
                TArgs.num_grad_accumalation_steps
            )
            grads = jax.tree_util.tree_map(
                lambda g1, g2: jnp.add(g1, g2), grads, grad
            ) # sum grads for grad accumation
            X_batch, y_batch = next(train_iterator)
        grad = None # save memory

        loss = loss*TArgs.num_grad_accumalation_steps # loss from last mini-step
        
        trainable_vars, (decay_opt_vars, nodecay_opt_vars) = update_params(
            grads, trainable_vars, (decay_opt_vars, nodecay_opt_vars)
        )
        grads = None # save memory

        if step % TArgs.eval_freq == 0 or step == TArgs.num_steps:
            print("Estimating Losses...")
            mean_losses, mean_accuracies = evaluate((trainable_vars, non_trainable_vars))
            print(
                f"\t| Training Loss: {mean_losses[0]:.4f} || Training Accuracy: {mean_accuracies[0]:.4f} |" 
                f"| Validation Loss: {mean_losses[1]:.4f} || Validation Accuracy: {mean_accuracies[1]:.4f} |"
            )
            
            _ = save_object(
                TArgs.checkpoint_dir+f"|checkpoint42M|gpt",
                obj=best_ckpt
            )
            print(f"Saved checkpoint of step {step}.")

            if mean_losses[1] < best_val_loss:
                best_val_loss = mean_losses[1]
                best_ckpt = (
                    trainable_vars, 
                    non_trainable_vars, 
                    (decay_opt_vars, nodecay_opt_vars),
                    step,
                    best_val_loss,
                    # training_losses
                )
                best_step = step
                wait = 0
            else:
                wait += 1

        # time
        t1 = time.time()
        dt = t1-t0; t0 = t1

        # print the essentials
        print(
            f"| Step: {step+start_iter} || Loss: {loss:.4f} || Accuracy: {accuracy:.4f} |"
            f"| LR: {learning_rate(step):e} || dt: {dt*1000:.2f}ms |"
        )
        # training_losses["train"].append(loss.tolist())
        step += 1

    train_state = (trainable_vars, non_trainable_vars)
    if TArgs.return_best_train_states:
        print(f"Best Weights are from Step {best_step}")
        print("With an Estimated Validation Loss of", best_val_loss)
        train_state = best_ckpt[:2] 
    # clear cell output, too large

    tstate_path = save_object(
        TArgs.checkpoint_dir+f"|train_state_42M|gpt",
        obj=train_state
    )
    print("Training States Saved at:", tstate_path)
    print("Done!")

    # plt.plot(training_losses["train"])
    # plt.title("Training Loss over Number of Steps")
    # plt.xlabel("Steps")
    # plt.ylabel("Train Loss")
    # plt.xticks(range(0, TArgs.num_steps+3_000, 3_000), rotation=90)
    # plt.yticks(jnp.arange(0, 11, 0.4).tolist())
    # plt.grid(True)
    # plt.savefig("train_loss_metrics.png")
    # plt.show()

if __name__ == "__main__":
    main()