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

import pickle

if __name__ == "__main__":
    
    ## Define problem setting:

    rng = RNG(64)

    n_subject = 32
    n_relation = 32
    n_answer_per = 8
    n_answer = n_answer_per*n_relation
    n_noise = 64
    vocab_size = n_subject + n_relation + n_noise+1

    T = 32

    fact_dict = jnp.array([jr.choice(rng.next(), jnp.arange(n_answer_per*i, n_answer_per*(i+1)), (n_subject,)) for i in range(n_relation)]).T

    problem = FactRecall(n_subject, n_relation, n_answer, n_noise, fact_dict,T,alpha=0., EOS_token=True)
    
    
    criterion = lambda f, y: -jnp.log(f[y])

    @jit
    def loss_fn(p, batch):
        x, y = batch
        return vmap(criterion)(f(p, x), y).mean()

    @jit
    def accuracy(p, batch):
        x, y = batch
        return vmap(lambda fi, yi: jnp.argmax(fi) == yi)(f(p, x), y).mean()
    
    ds = [32]
    dh = 8
    Hs = [1,  2,  4,  6,  8, 10, 12, 14, 16, 20, 24, 28, 32]
    ms = [1,  8,  16,  24,  32,  40,  48,  56,  64,  80,  96, 112, 128]
    
    
    print("Training")
    lr = 1e-2
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
                
    # results = [[0. for H in Hs] for d in ds]
    results_dict = {}

    for k in range(len(ds)):
        for j in range(len(Hs)):
            for l in range(len(ms)):
                d = ds[k]
                H = Hs[j]
                m = ms[l]
                
                ## Init model
                model = TF_one_layer(vocab_size=vocab_size, output_size=n_answer, d=d, heads=H, d_h = dh, width=m, alpha=1.)

                p0 = model.init(rng.next(), vmap(problem.sample)(rng.next(2))[0])
                p0 = flatten_dict(p0["params"], sep=".")

                @partial(jit, static_argnames="mutable")
                def f(p, *args, **kwargs):
                    p = dict(params=unflatten_dict(p, sep="."))
                    return model.apply(p, *args, **kwargs)
                
                
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
                        if acc == 1:
                            break
                    batch = next(iterator)
                    p, opt_state, loss = step_fn_adam(p, batch, opt_state)
                    train_loss.append(loss)
                test_accs.append(accuracy(p, (testx, testy)))
                acc = max(test_accs)
                # results[k][j] = acc.item()
                results_dict[(d, H, dh, m)] = acc.item()
                print(d, H, dh, m, acc)
            
    with open('all_results.pkl','wb') as f:
        pickle.dump(results_dict, f)
    