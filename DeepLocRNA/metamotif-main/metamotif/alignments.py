# %%
import numpy as np

# %%
class SeededMotifAlignment:
    def __init__(self, seed, size=5, min_agreement=3):
        # params
        self.size = size
        self.min_agreement = min_agreement
        self.padding = self.size - self.min_agreement
        self.extended_size = self.size + 2 * self.padding
        
        # seed
        self.seed = seed
        self.padded_seed = np.zeros(shape=(self.extended_size, 4), dtype=np.float32)
        self.padded_seed[self.padding:(-self.padding)] = self.seed
        
        # pam
        self.pam = np.zeros(shape=(self.extended_size, 4), dtype=np.float32)
        self.pam[self.padding:(-self.padding)] = self.seed
        # self.support = np.zeros(shape=(self.extended_size, ))
        # self.support[self.padding:(-self.padding)] = np.ones(shape=(self.size, ))
        
        self.support = 1
    
    @property
    def pwm(self):
        return self.pam / self.support
    
    def align(self, kmer):
        assert len(kmer) == self.size
        
        agreement_max_idx, agreement_max_val = None, 0
        for i in range(self.extended_size - self.size + 1):
            agreement_i_val = np.sum(kmer * self.padded_seed[i:(i+self.size)])
            if agreement_i_val > agreement_max_val:
                agreement_max_val = agreement_i_val
                agreement_max_idx = i
                
        if agreement_max_val < self.min_agreement:
            return False
        else:
            self.pam[agreement_max_idx:(agreement_max_idx + self.size)] += kmer
            self.support += 1
            return True

# %%
