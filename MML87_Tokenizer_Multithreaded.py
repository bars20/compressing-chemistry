import numpy as np
from tqdm import tqdm
import pickle
import random
import pickle
from joblib import Parallel, delayed
import math
import re
import copy
from time import perf_counter as pf
class MML_Tokenizer:
    def __init__(self, data, reserved_chars=[], max_len = 4, prior = 'relative_frequency', syntax_check = False, cores = 8):
        '''
        data: list of lists of symbols
        '''
        self.ncores=cores
        self.data = [list(i) for i in data] #ensure data is a list
        self.chars = set(word for sentence in data for word in sentence)#determine the characters in the dataset
        self.sym_dic = {i:j for i,j in zip(reserved_chars, np.arange(len(reserved_chars)))} #make symbol dictionaries for the characters
        self.max_int = len(self.chars) + len(reserved_chars)
        l = len(reserved_chars)
        self.sym_dic[' '] = l
        l+=1
        for char in self.chars:
            #add the alphabet first, then brackets 
            self.sym_dic[char] = l
            l += 1
        self.chars = self.chars.union(set(reserved_chars)) 
        self.num_chars = [i for i in self.chars if any(j in '0123456789' for j in i) and len(i) > 1]
        self.rev_sym_dic = {j:i for i,j in self.sym_dic.items()}
        #map the data to lists of integers
        self.encoded_data = [[self.sym_dic[char] for char in word] for word in self.data]
        #initialise the rule dictionaries
        self.rule_dic = dict() #map from tuples of characters to new integers
        self.r_rule_dic = dict() #reverse above
        self.unrolled_rule_dic = dict() #like self.rule_dic, but tuple of characters must be in terms of integers in the original symbols
        self.r_unrolled_rule_dic = dict() #reverse above
        self.max_pair, self.codelen = None, None
        self.max_len = max_len
        #determine the codelengths assigned to the original symbols in the theory part of the message
        org_counts = dict()
        for i in self.encoded_data:
            for j in i:
                org_counts[j] = org_counts.get(j,0) + 1
        if prior == 'relative_frequency':
            self.org_codelens = {(i,):-np.log(org_counts[i]/sum(org_counts.values())) for i,_ in org_counts.items()}
        else:
            self.org_codelens = {(i,):-np.log(1/len(org_counts)) for i,_ in org_counts.items()}
        self.counts = None
        self.sub_scores = dict()
        # if syntax_check == False:
        #     self.syntax_check = lambda x: True
        # else:
        #     self.syntax_check = self.syntax_check_chem

    def add_one(self,x ):
        return x+ 1

    def syntax_check(self, subword):
        if len(subword) == 1:
            return True
        if '.' in subword or 'Y' in subword:
            return False
        for i in '0123456789':
            if i in subword:
                if subword.count(i)%2 != 0:
                    return False
                if subword.index(i) == 0 or  subword.index(i) == len(subword)-1:
                    return False
        #now check for matching brackets:
        for a1, a2 in zip('([',')]'):
            if a1 in subword and a2 not in subword:
                return False
            if a2 in subword and a1 not in subword:
                return False
            elif subword.count(a1) != subword.count(a2):
                return False 
            #ensure that ( occurs before ) for all matching brackets
            else:
                for bracket in ['()','[]']:
                    b1, b2 = bracket
                    o = 0
                    c = 0
                    for char in subword:
                        if char == b1:
                             if c > o:
                                return False
                             else:
                                o += 1
                        elif char == b2:
                            if c > o:
                                return False
                            else:
                                c += 1      
        for i in '=#-':
            if i in subword:
                idx = subword.index(i)
                if idx == 0 or idx == len(subword) -1:
                    return False
                elif not subword[idx-1].isalpha() and not subword[idx+1].isalpha():
                    return False
        return True
    
    def can_overlap(self, seq):
        '''Determine whether a sequence can overlap with itself
        '''
        mx = len(seq) // 2
        for i in range(1,mx+1):
            if seq[:i] == seq[-i:]:
                return True
        return False

    def max_no(self, sub, idxs):
        idx = 0
        l=len(sub)
        while idx < len(idxs):
            idxs = list(idxs[:idx+1]) + list(filter(lambda x: x >= l+idxs[idx], idxs))
            idx += 1
        return idxs

    def remove_invalid_subs(self, dic):
        return {i:j for i,j in dic.items() if self.syntax_check(self.decode_sentence([list(i)])[:-1])}

    def count_sublists_limited(self, word, total_count_dic, mx):
        idx_dic = dict()
        idx1 = 0
        idx2 = 1
        while idx1 < len(word):
            while idx2 < idx1+mx+1 and idx2<len(word)+1:
                subword = word[idx1:idx2]
                if True: #self.syntax_check(self.decode_sentence([subword])[:-1]):
                    if tuple(subword) in idx_dic.keys():
                        idx_dic[tuple(subword)] = idx_dic[tuple(subword)] + (idx1,)
                    else:
                        idx_dic[tuple(subword)] = (idx1,)
                idx2 += 1
            idx1 += 1
            idx2 = idx1 + 1
        #then determine the max counts of each subword
        for sub, idxs in idx_dic.items():
            #determine if the sub is a candidate for overlap
            if not self.can_overlap(sub):
                no = len(idxs)
                if sub in total_count_dic.keys():
                    total_count_dic[sub] += no
                else:
                    total_count_dic[sub] = no
            else:
                pruned = self.max_no(sub,idxs)
                no = len(pruned)
                if sub in total_count_dic.keys():
                    total_count_dic[sub] += no
                else:
                    total_count_dic[sub] = no
        return total_count_dic
    
    def count_sublists_limited_p(self, word, total_count_dic = dict()):
        mx = self.max_len
        idx_dic = dict()
        idx1 = 0
        idx2 = 1
        while idx1 < len(word):
            while idx2 < idx1+mx+1 and idx2<len(word)+1:
                subword = word[idx1:idx2]
                if True: #self.syntax_check(self.decode_sentence([subword])[:-1]):
                    if tuple(subword) in idx_dic.keys():
                        idx_dic[tuple(subword)] = idx_dic[tuple(subword)] + (idx1,)
                    else:
                        idx_dic[tuple(subword)] = (idx1,)
                idx2 += 1
            idx1 += 1
            idx2 = idx1 + 1
        #then determine the max counts of each subword
        for sub, idxs in idx_dic.items():
            #determine if the sub is a candidate for overlap
            if not self.can_overlap(sub):
                no = len(idxs)
                if sub in total_count_dic.keys():
                    total_count_dic[sub] += no
                else:
                    total_count_dic[sub] = no
            else:
                pruned = self.max_no(sub,idxs)
                no = len(pruned)
                if sub in total_count_dic.keys():
                    total_count_dic[sub] += no
                else:
                    total_count_dic[sub] = no
        return total_count_dic
    
    def count_words(self, words, dic=dict()):
        for word in words:
            dic = self.count_sublists_limited(word, dic, self.max_len)
        return dic
    
    def safe_I_1(self, N,M,s):
        corrective_factor = math.lgamma(N+1) - np.sum([math.lgamma(i) for i in np.add(s, 1)])
        return corrective_factor + math.lgamma(N+M) - math.lgamma(M) - np.sum([math.lgamma(i) for i in np.add(s, 1)]) + 0.5*np.log((M-1)*math.pi) - 0.4

    def calc_assertion(self, dic=None,syms=[]):
        #rewrite this to be shorter - quick fix for now
        if dic == None:
            self.prune_dic()
            total = 0
            for sub in [i for i in list(self.pruned_dic.keys()) + syms]: 
                unrolled = self.unroll_pair(sub)
                #determine cost of asserting the unrolled substructure
                total += (self.logstar(len(unrolled)) + sum([self.org_codelens[(s,)] for s in unrolled]))
        else:
            total = 0
            for sub in dic.keys():
                unrolled = self.unroll_pair(sub)
                #determine cost of asserting the unrolled substructure
                total += (self.logstar(len(unrolled)) + sum([self.org_codelens[(s,)] for s in unrolled]))
        return total

    def codelength(self):
        sym_counts = dict()
        for seq in self.encoded_data:
            sym_counts = self.count_sublists_limited(seq, sym_counts, 1)
        assertion_len = self.calc_assertion(dic=sym_counts)
        N = sum(sym_counts.values())
        M = len(sym_counts)
        MML87_codelen = self.safe_I_1(N, M, list(sym_counts.values())) + self.logstar(N) + self.logstar(M)
        return assertion_len + MML87_codelen

    def split_data(self, data):
        chunk_sz = int(np.ceil(len(data) / self.ncores))
        return [data[chunk_sz*i: chunk_sz*(i+1)] for i in range(math.ceil(len(data) / chunk_sz))]
    
    def get_max_score(self, cores=8):
        #split data and count subwords in parallel
        split_d = self.split_data(self.encoded_data) 
        parallel = Parallel(n_jobs=self.ncores, return_as="list")
        dics = parallel(delayed(self.count_words) (s, dict()) for s in split_d)
        #join and add resulting dictionaries
        cand_dic = {k: sum(t.get(k, 0) for t in dics) for k in set.union(*[set(t) for t in dics])}
        cand_dic = self.remove_invalid_subs(cand_dic)
        #split dictionary and score each subword in parallel
        initial_counts = {i:j for i,j in cand_dic.items() if len(i) == 1} #all symbols
        res = self.split_data(list(cand_dic.items())) #split candidate dictionary
        outs  = parallel(delayed(self.score_subs)((i, initial_counts)) for i in res) #score each substructure
        sub_score_dics = list(map(lambda x: x[2], outs)) #score dictionary
        sub_scores = {k: sum(t.get(k, 0) for t in sub_score_dics) for k in set.union(*[set(t) for t in sub_score_dics])} #combine dictionaries
        if len(sub_scores) == 0: #if there are no substructures to score, return None
            return None
        return min(sub_scores.items(), key=lambda x:x[1])
    
    def score_subs(self, comb):
        cand_dic, init_counts = comb
        cand_dic  = {i:j for i,j in cand_dic}
        MML_87 = dict()
        as_dic = dict()
        sub_scores = dict()
        #takes in some part of the candidate dictionary and calculates scores for each, return all dictionaries
        for sub in cand_dic.keys():
            if len(sub) > 1:
                new_counts = init_counts.copy() #this is all symbols
                new_counts[sub] = cand_dic[sub] #add sub to this dictionary
                t_dic = self.count_sublists_limited(sub, dict(), 1) #get counts of all symbols in sub
                for sym in t_dic:
                    new_counts[sym] -= (new_counts[sub] * t_dic[sym])
                new_counts = {i:j for i,j in new_counts.items() if j > 0}
                N = sum(new_counts.values())
                M = len(new_counts)
                MML_87[sub] = self.safe_I_1(N,M,list(new_counts.values())) + self.logstar(N) + self.logstar(M)
                as_dic[sub] = self.calc_assertion(dic=new_counts)
                sub_scores[sub] = MML_87[sub] + as_dic[sub]
        return MML_87, as_dic, sub_scores
    
    def unroll_pair(self, pair):
        #if the pair is a tuple containing only a symbol, return the symbol as a tuple
        if len(pair) == 1 and pair[0] < len(self.chars):
            return pair
        unrolled = tuple()
        for num in pair:
            if num <= len(self.chars):
                #find a rule in the rule dictionary that has this number
                unrolled += (num,)
            else:
                new_num = self.r_rule_dic[num]
                unrolled += self.unroll_pair(new_num)
        return unrolled
            
    
    def encode(self):
        if len(self.encoded_data) == 1 and len(self.encoded_data[0]) == 1: #don't encode if there is only one word
            return None, self.codelen
        #first get the max pair from the dataset 
        score = self.get_max_score()
        if score != None:
            max_seq, self.codelen = score
            l = len(max_seq)
        else:
            return None, self.codelen
        #encode data in parallel
        pieces = self.split_data(self.encoded_data)
        #prepare dictionaries for new word
        self.rule_dic[max_seq] = self.max_int+1 
        self.r_rule_dic[self.max_int+1] = max_seq
        unrolled_pair = self.unroll_pair(max_seq)
        self.unrolled_rule_dic[unrolled_pair] = self.max_int+1 
        self.r_unrolled_rule_dic[self.max_int+1] = unrolled_pair
        parallel = Parallel(n_jobs=self.ncores, return_as="list")
        res = parallel(delayed(self.encode_words)((p, max_seq, l)) for p in pieces)
        encoded_data = sum(res, []) #combine lists
        self.max_int += 1
        self.encoded_data = encoded_data #update class variable
        return max_seq, self.codelen

    def encode_words(self, packed):
        words, max_seq, l = packed
        encoded_data = []
        for word in words: 
            w_idx = 0
            encoded_word = []
            while w_idx < len(word):
                if w_idx < len(word) and tuple(word[w_idx:w_idx+l]) == max_seq:
                    encoded_word.append(self.max_int+1)
                    w_idx += l
                else:
                    encoded_word.append(word[w_idx])
                    w_idx +=1
            encoded_data.append(encoded_word)
        return encoded_data
    
    def get_used_tokens(self, data):
            return set([token for sentence in data for token in sentence])

    def prune_dic(self):
        #check if item occurs in encoded data
        used_tokens = set([token for sentence in self.encoded_data for token in sentence])
        #print(used_tokens)
        t_dic = dict()
        for token in self.rule_dic.values():
            if token > len(self.chars) and token in used_tokens:
                #add the rule for this token to the rule dictionary
                t_dic[token] = self.r_rule_dic[token]
            # if token in used_tokens:
            #     t_dic[token] = self.rule_dic[token]
        #THEN NEED TO SHORTEN USING METHOD
        #print('done')
        self.pruned_dic = {j:i for i,j in t_dic.items()}
        
    def logstar(self, n):
        if n < 2:
            return 1
        # return self.head(n) + 1
        else:
            return np.log(n) + np.log(np.log(n))

    def decode_sym(self,symbol):
        if symbol <= len(self.chars):
            return self.rev_sym_dic[symbol]
        else:
            return ''.join(self.decode_sym(el) for el in self.r_rule_dic[symbol])
    
    def decode_sentence(self,sentence):
            self.r_rule_dic = {j:i for i,j in self.rule_dic.items()} 
            decoded = ''
            for word in sentence:
                for symbol in word:
                    decoded += self.decode_sym(symbol)
                decoded += ' '
            return decoded
