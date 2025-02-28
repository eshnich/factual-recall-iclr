import jax
from jax import nn
from jax import numpy as jnp
from jax import random as jr
from simple_pytree import Pytree, static_field
from jax.tree_util import register_pytree_node_class
from jax import numpy as jnp
from flax import linen as nn
from flax.linen import initializers as nni

class Transformer(nn.Module):
    vocab_size: int
    output_size: int
    d: int
    heads: int
    width: int # width of MLP
    
    def attn(self, x, A):
        T = x.shape[-2]
        attn = jnp.einsum("...ij,jk,...lk -> ...il", x, A, x)
        attn = jnp.where(jnp.tri(T), attn, -jnp.inf)
        attn = nn.softmax(attn)
        attn = jnp.einsum("...ij,...jk->...ik", attn, x)
        return attn
    
    def embed(self, x, wte):
        out = wte[x]
        return out
    
    @nn.compact
    def __call__(self, x):
        A = self.param('A', nni.normal(0.1), (self.heads, self.d, self.d))
        V = self.param('V', nni.zeros, (self.heads*self.d, self.d))
        
        # # fixing V to prevent self-attention from memorizing!
        # T = jnp.einsum("i,jk -> ijk", jnp.ones(self.heads), jnp.eye(self.d))
        # V = T.reshape(-1, self.d)
        
        wte = self.param('wte', nni.normal(1./jnp.sqrt(self.d)), (self.vocab_size, self.d))
        unembed = self.param('unembed', nni.normal(1./jnp.sqrt(self.d)), (self.output_size, self.d))
        
        x = self.embed(x, wte)
    
        attn = jax.vmap(self.attn, (None, 0), -2)(x, A)
        attn = attn.reshape(*attn.shape[:-2], -1)
        attn = attn@V
        x = attn[..., -1, :]
        
        z = nn.Dense(self.width, name = 'layer1')(x)
        z = nn.relu(z)
        z = nn.Dense(self.d, use_bias = False, kernel_init = nni.zeros, name = 'layer2')(z)
        x = z + x
        return nn.softmax(x@unembed.T)
    
## Standard parameterization    
class TF_one_layer(nn.Module):
    vocab_size: int
    output_size: int
    d: int
    heads: int
    d_h: int # head dimension
    width: int # width of MLP
    alpha: int # how much to weight the MLP
    init_scale: int = 1
    
    def attn(self, x, Q, K, V):
        attn = jnp.einsum("...ij,jk,...k -> ...i", x, K@Q.T, x[...,-1,:])
        attn = nn.softmax(attn/jnp.sqrt(self.d_h))
        attn = jnp.einsum("...ij,...i->...j", x@V, attn)
        return attn
    
    def embed(self, x, wte):
        out = wte[x]
        return out
    
    def attn_only(self, x, wte, unembed, Q, K, V, O):
        
        x = self.embed(x, wte)
        attn = jax.vmap(self.attn, (None, 0, 0, 0), -2)(x, Q, K, V)
        attn = attn.reshape(*attn.shape[:-2], -1)
        x = attn@O
        
        return nn.softmax(x@unembed.T)
    
    @nn.compact
    def __call__(self, x):
        Q = self.param('Q', nni.normal(self.init_scale/jnp.sqrt(self.d_h)), (self.heads, self.d, self.d_h))
        K = self.param('K', nni.normal(self.init_scale/jnp.sqrt(self.d_h)), (self.heads, self.d, self.d_h))
        V = self.param('V', nni.normal(self.init_scale/jnp.sqrt(self.d_h)), (self.heads, self.d, self.d_h))
        O = self.param('O', nni.normal(self.init_scale/jnp.sqrt(self.d)), (self.heads*self.d_h, self.d))
        
        wte = self.param('wte', nni.normal(1./jnp.sqrt(self.d)), (self.vocab_size, self.d))
        unembed = self.param('unembed', nni.normal(1./jnp.sqrt(self.d)), (self.output_size, self.d))
        
        x = self.embed(x, wte)
    
        attn = jax.vmap(self.attn, (None, 0, 0, 0), -2)(x, Q, K, V)
        attn = attn.reshape(*attn.shape[:-2], -1)
        x = attn@O
        
        z = nn.Dense(self.width, name = 'layer1')(x)
        z = nn.relu(z)
        z = nn.Dense(self.d, use_bias = False, name = 'layer2')(z)
        x = self.alpha*z + x
        return nn.softmax(x@unembed.T)