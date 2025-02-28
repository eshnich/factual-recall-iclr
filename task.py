from jax import numpy as jnp
from jax import random as jr

class FactRecall:
    def __init__(self, n_subject, n_relation, n_answer, n_noise, fact_dict, T, alpha = 0., beta = 0., EOS_token = False):
        self.n_subject = n_subject 
        self.n_relation = n_relation
        self.n_answer = n_answer
        self.n_noise = n_noise
        self.T = T
        self.dict = fact_dict
        p = 1./jnp.arange(1, n_subject + 1)**alpha
        self.p = p/p.sum()
        q = 1./jnp.arange(1, n_relation + 1)**beta
        self.q = q/q.sum()
        # print(self.p)
        self.EOS = EOS_token
        
    def sample(self, key):
        skey, lkey, rkey, lrkey, nkey = jr.split(key, 5)
        subject = jr.choice(skey, self.n_subject, p=self.p)
        location = jr.choice(lkey, self.T-1)
        relation = jr.choice(rkey, self.n_relation, p = self.q)
        r_location = jr.choice(lrkey, self.T-2)
        r_location = r_location + (r_location >=location)
        x = jr.choice(nkey, jnp.arange(self.n_subject + self.n_relation, 
                                           self.n_subject + self.n_relation + self.n_noise), (self.T,))
        x = x.at[location].set(subject)
        if self.EOS:
            x = x.at[r_location].set(relation + self.n_subject)
            x = x.at[self.T-1].set(self.n_subject + self.n_relation + self.n_noise)
        else:
            x = x.at[self.T-1].set(relation + self.n_subject)
        
        answer = self.dict[subject, relation]
        
        return x, answer