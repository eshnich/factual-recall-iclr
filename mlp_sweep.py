#!/bin/bash

import jax
from jax import nn
from jax import numpy as jnp
from jax import random as jr
from util import *
import optax
from flax.traverse_util import flatten_dict, unflatten_dict

import pickle
import argparse

## Standard parameterization    
class MLP(nn.Module):

    d: int
    width: int # width of MLP
    
    @nn.compact
    def __call__(self, x):
        z = nn.Dense(self.width, name = 'layer1')(x)
        z = nn.relu(z)
        # z = z**2 + z**3
        z = nn.Dense(self.d, use_bias = False, name = 'layer2')(z)
        return z

def run(d, m, N, lr, rng, M):
    
    lr = lr
    wd = 0.
    steps = 2**15
    save_every = steps // 128
    # M = size of output vocabulary
    
    # U = jr.normal(rng.next(), (N, d))
    U = jr.normal(rng.next(), (M, d))
    U = vmap(lambda u: u/jnp.linalg.norm(u))(U)
    E = jr.normal(rng.next(), (N, d))
    E = vmap(lambda u: u/jnp.linalg.norm(u))(E)
    
    model = MLP(d = d, width = m)

    p0 = model.init(rng.next(), E)
    p0 = flatten_dict(p0["params"], sep=".")
    
    
    @partial(jit, static_argnames="mutable")
    def f(p, *args, **kwargs):
        p = dict(params=unflatten_dict(p, sep="."))
        return model.apply(p, *args, **kwargs)
    @jit
    def criterion(i, y_hat):
        return -jnp.log(y_hat[i])
    
    f_star = jnp.array([int(i*M/N) for i in jnp.arange(N)])

    @jit
    def loss_fn(p):
        preds = nn.softmax(f(p, E) @ U.T)
        return vmap(criterion)(f_star, preds).mean()

    @jit
    def accuracy(p):
        preds = f(p, E) @ U.T
        return vmap(lambda yi, fi: jnp.argmax(fi) == yi)(f_star, preds).mean()
    
    
    # Init optimizer
    opt = optax.adamw(learning_rate = lr, weight_decay=wd)

    @jit
    def step_fn_adam(p, opt_state):
        loss, g = jax.value_and_grad(loss_fn)(p)
        updates, opt_state = opt.update(g, opt_state, p)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss
    
    p = p0
    losses = []
    accs = []
    
    opt_state = opt.init(p0)
    for i in trange(steps):
        p, opt_state, loss = step_fn_adam(p, opt_state)
        losses.append(loss)
        if i % save_every == 0:
            acc = accuracy(p)
            accs.append(acc)
            if acc > 0.99:
                break

    acc = max(accs)
    print(acc)
    return acc

def binary_sweep(d, m, min_N, max_N, lrs, rng, M, fixed_output):
    
    max_size = max_N
    min_size = min_N
    while (max_size/min_size > 1.01) and (max_size - min_size > 1):
        N = int(0.5*(max_size + min_size))
        N_out = M if fixed_output else N
        
        for lr in lrs:
            acc = run(d, m, N, lr, rng, N_out)
            print(N, lr, acc)
            if acc >= 0.99:
                min_size = N
                break
        if acc < 0.99:
            max_size = N
            
    return min_size

if __name__ == "__main__":
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some arguments.')

    # Add an argument (e.g., arg)
    parser.add_argument('--alpha', type=int, required=True, help='alpha') #m/d ratio
    parser.add_argument('--seed', type=int, required=True, help='seed')

    # Parse the arguments
    args = parser.parse_args()
    
    alpha = args.alpha
    seed = args.seed
    print("alpha = {}".format(alpha))
    print(jax.devices())
    
    d_range = [int(2**i) for i in jnp.arange(3, 7, 0.5)]
    rng = RNG(seed)
    results = []
    
    M = 32 # fixed number of output classes
    fixed_output = False # do we use a fixed (M) number of output classes, or let it scale with N?
    
    for d in d_range:
        print("training d = {}".format(d))
        N = binary_sweep(d, alpha*d, d, alpha*d*d, [3e-2], rng, M, fixed_output)
        results.append((alpha*d*d, N))
    
    if fixed_output:
        filename = 'mlp_sweep_alpha={}_M={}_seed={}.pkl'.format(alpha, M, seed)
    else:
        filename = 'mlp_sweep_alpha={}_seed={}.pkl'.format(alpha, seed)
        
    with open(filename,'wb') as f:
        pickle.dump(results, f) 