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

from run import run, sweep, binary_sweep

import pickle
import argparse

if __name__ == "__main__":
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some arguments.')

    # Add an argument (e.g., arg)
    parser.add_argument('--alpha', type=float, required=True, help='alpha')
    parser.add_argument('--beta', type=float, required=True, help='beta')
    parser.add_argument('--H', type=int, required=True, help='H')
    parser.add_argument('--D', type=int, required=True, help='D')

    # Parse the arguments
    args = parser.parse_args()
    
    alpha = args.alpha
    beta = args.beta
    H = args.H
    D = args.D
    print(alpha, beta, H)
    
    
    d_range = [int(2**i) for i in jnp.arange(3, 8, 0.5)]
    min_size = 8
    max_size = 512
    
    
    results= []
    
    # specify scaling of (S, R, D)
    # scaling = "square" # S = R^2
    # scaling = "distinct" # D = S
    scaling = "linear" # S = R
    
    for d in d_range:
        dh = int(d*alpha/H)
        m = int(d*beta)
        size = binary_sweep(H, dh, d, m, D, min_size, max_size, scaling, [1e-3, 3e-3, 1e-2])
        min_size = size
        
        if scaling == "linear" or scaling == "distinct":
            num_facts = size**2
        elif scaling == "square":
            num_facts = size*int(size**0.5)
            
        num_params = 4*H*dh*d + 2*m*d
        results.append((num_params, num_facts))
        
    
    if scaling == "linear":
        filename = 'sweep_alpha={}_beta={}_H={}_D={}.pkl'.format(alpha, beta, H, D)
    elif scaling == "square":
        filename = 'sweep_alpha={}_beta={}_H={}_D={}_large_S.pkl'.format(alpha, beta, H, D)
    elif scaling == "distinct":
        filename = 'sweep_alpha={}_beta={}_H={}_D=S.pkl'.format(alpha, beta, H)
    
    with open(filename,'wb') as f:
        pickle.dump(results, f) 
    
    
    