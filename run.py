#!/bin/bash

import jax
from jax import jit, lax
from jax import numpy as jnp
from jax import tree_util
from jax import random as jr
from jax import vmap
from jax.numpy import linalg as jla
from util import *
from simple_pytree import Pytree, static_field
import optax
from flax.traverse_util import flatten_dict, unflatten_dict

from task import FactRecall
from model import Transformer, TF_one_layer

from jax import numpy as jnp
from flax import linen as nn
from flax.linen import initializers as nni


# specify the model size (H, dh, d, m) and the problem size (S, R, D)
# outputs the accuracy
def run(H, dh, d, m, S, R, D, lr):
    
    rng = RNG(64)

    n_answer = D*R
    n_noise = S + R
    vocab_size = S + R + n_noise + 1

    T = 32

    fact_dict = jnp.array([jr.choice(rng.next(), jnp.arange(D*i, D*(i+1)), (S,)) for i in range(R)]).T

    problem = FactRecall(S, R, n_answer, n_noise, fact_dict,T,alpha=0., EOS_token=True)
    
    print("Training")
    lr = lr
    wd = 0.
    steps = 2**14
    save_every = steps // 128
    batch_size = 2**10
    max_size = 2**24
    epoch_len = max_size // batch_size
    sample_fn = jit(lambda k: vmap(problem.sample)(jr.split(k, epoch_len * batch_size)))

    testx, testy = vmap(problem.sample)(rng.next(2**10))
    
    def batch_iterator(key):
        while True:
            key, subkey = jr.split(key)
            batches = sample_fn(subkey)
            for i in range(epoch_len):
                yield tree_map(lambda x: x[batch_size * i : batch_size * (i + 1)], batches)
            
    ## Init model
    model = TF_one_layer(vocab_size=vocab_size, output_size=n_answer, d=d, heads=H, d_h = dh, width=m, alpha=1.)

    p0 = model.init(rng.next(), vmap(problem.sample)(rng.next(2))[0])
    p0 = flatten_dict(p0["params"], sep=".")

    criterion = lambda f, y: -jnp.log(f[y])
    
    @partial(jit, static_argnames="mutable")
    def f(p, *args, **kwargs):
        p = dict(params=unflatten_dict(p, sep="."))
        return model.apply(p, *args, **kwargs)
    
    @jit
    def loss_fn(p, batch):
        x, y = batch
        return vmap(criterion)(f(p, x), y).mean()

    @jit
    def accuracy(p, batch):
        x, y = batch
        return vmap(lambda fi, yi: jnp.argmax(fi) == yi)(f(p, x), y).mean()
    
    
    # Init optimizer
    opt = optax.multi_transform(
        {'adam': optax.adamw(learning_rate = lr, weight_decay=wd),
        'zero': optax.set_to_zero()
        },
        {
            'wte':'zero',
            'unembed':'zero',
            'Q':'adam',
            'K':'adam',
            'V':'adam',
            'O':'adam',
            'layer1.kernel': 'adam',
            'layer1.bias': 'adam',
            'layer2.kernel': 'adam'
        })

    @jit
    def step_fn_adam(p, batch, opt_state):
        loss, g = jax.value_and_grad(loss_fn)(p, batch)
        updates, opt_state = opt.update(g, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    iterator = batch_iterator(rng.next())
    
    p = p0
    train_loss = []
    test_loss = []
    test_accs = []
    
    opt_state = opt.init(p0)
    for i in trange(steps):
        if i % save_every == 0:
            test_loss.append(loss_fn(p, (testx, testy)))
            acc = accuracy(p, (testx, testy))
            test_accs.append(acc)
            if acc > 0.99:
                break
        batch = next(iterator)
        p, opt_state, loss = step_fn_adam(p, batch, opt_state)
        train_loss.append(loss)
    test_accs.append(accuracy(p, (testx, testy)))
    acc = max(test_accs)
    print(acc)
    return acc

def sweep(H, dh, d, m, D, SR_list):

    final_size = 0
    for (S, R) in SR_list:
        acc = run(H, dh, d, m, S, R, D)
        if acc < 0.99:
            break
        final_size = S*R


    return final_size

def binary_sweep(H, dh, d, m, D, min_SR, max_SR, scaling, lrs):
    
    max_size = max_SR
    min_size = min_SR
    while max_size - min_size > 1:
        
        size = int(0.5*(max_size + min_size))
        
        for lr in lrs:
            
            if scaling == "linear":
                R = size
                D_new = D
            elif scaling == "square":
                R = int(size**0.5)
                D_new = D
            elif scaling == "distinct":
                R = size
                D_new = size
                
            acc = run(H, dh, d, m, size, R, D_new, lr)
            print(size, lr, acc)
            if acc >= 0.99:
                min_size = size
                break
        if acc < 0.99:
            max_size = size
            
    
    return min_size
            
    