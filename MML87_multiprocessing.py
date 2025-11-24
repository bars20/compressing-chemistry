#Multiprocessing Script
import numpy as np
import multiprocessing
from multiprocessing import Pool
from time import perf_counter as pf
from MML87_Tokenizer_Multithreaded import MML_Tokenizer
import math
import pickle
import copy
import pandas as pd
from rdkit import Chem
import re
from tqdm import tqdm
import os
#parallel functions on the library 
class Parallel_Functions:
    def __init__(self):
        pass

    def remove_invalid_subs(self, dic):
        return {i:j for i,j in dic.items() if self.syntax_check(self.decode_sentence([list(i)])[:-1])}
    
    def decode_sym(self, symbol):
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
    
    def syntax_check(self, subword):
        if len(subword) == 1:
            return True
        if '.' in subword or 'Y' in subword:
            return False
        rem_sym = None
        patt = None
        for i in '0123456789':
            if i in subword:
                if subword[0].isnumeric():
                    return False
                if patt == None and rem_sym == None:
                    #check number of symbols chars
                    patt = ''.join([re.escape(i)+ '|' for i in self.num_chars])[:-1]
                    rem_sym, _ = re.subn(patt,'X', subword) #get number of symbol characters
                if rem_sym.count(i) % 2 != 0:
                    return False
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
        
    def get_used_tokens(self, data):
            return set([token for sentence in data for token in sentence])
    
    def count_sublists_limited(self, word, total_count_dic = dict(), mx=8):
        #mx = self.max_len
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
    
    def count_words(self, words):
        dic = dict()
        for word in words:
            dic = self.count_sublists_limited(word, total_count_dic=dic, mx=self.max_len)
        return dic
    

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
    #12534 35
    def score_subs_old(self, comb):
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
                t_dic = self.count_sublists_limited(sub, total_count_dic=dict(), mx=1) #get counts of all symbols in sub
                #print(sub, t_dic)
                #one of symbols in sub is not in initial counts:  HOW??
                for sym in t_dic:
                    new_counts[sym] -= (new_counts[sub] * t_dic[sym])
                new_counts = {i:j for i,j in new_counts.items() if j > 0}
                N = sum(new_counts.values())
                M = len(new_counts)
                #print(N,M, sum(init_counts.values()))
                MML_87[sub] = self.safe_I_1(N,M,list(new_counts.values())) + self.logstar(N) + self.logstar(M)
                as_dic[sub] = self.calc_assertion(dic=new_counts)
                sub_scores[sub] = MML_87[sub] + as_dic[sub]
                #print(as_dic[sub], sub_scores[sub])
        return MML_87, as_dic, sub_scores
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
                t_dic = self.count_sublists_limited(sub, total_count_dic=dict(), mx=1) #get counts of all symbols in sub
                for sym in t_dic:
                    new_counts[sym] -= (new_counts[sub] * t_dic[sym])
                #s1 = pf()
                #new_counts = {i:j for i,j in new_counts.items() if j > 0} #at this step determine the difference from prev dic? store old values?
                new = dict()
                rm = dict()
                for i,j in new_counts.items():
                    if j > 0:
                        new[i] = j
                    else:
                        rm[i] = j
                new_counts = new
                N = sum(new_counts.values())
                M = len(new_counts)

                #print(N,M, sum(init_counts.values()))
                MML_87[sub] = self.safe_I_1(N,M,list(new_counts.values())) + self.logstar(N) + self.logstar(M)
                #maybe only feed in new substructure and removed, and adjust existing assertion length
                if self.total != None:
                    as_dic[sub] = self.alter_ass(sub, rm)
                    # old = self.calc_assertion(dic=new_counts)
                    # if round(as_dic[sub], 4) != round(old,4):
                    #     print(sub, 'issue', as_dic[sub], old)
                else:
                    as_dic[sub] = self.calc_assertion(dic=new_counts)
                #as_dic[sub] = self.calc_assertion(dic=new_counts)
                sub_scores[sub] = MML_87[sub] + as_dic[sub]
                #e3 = pf()
                #all_sub_times.append([self.decode_sentence([list(sub)])[:-1], [e1-s1, e2-s2, e3-s3]])
        #print(sorted(all_sub_times, key=lambda x:sum(x[1]), reverse=True)[:10])
                #print(as_dic[sub], sub_scores[sub])
        return MML_87, as_dic, sub_scores
    
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

    def alter_ass(self, add, rem):
        total = self.total
        for sub in rem.keys():
            unrolled = self.unroll_pair(sub)
            total -= (self.logstar(len(unrolled)) + sum([self.org_codelens[(s,)] for s in unrolled]))
        unrolled = self.unroll_pair(add)
        total += (self.logstar(len(unrolled)) + sum([self.org_codelens[(s,)] for s in unrolled]))
        return total


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

    def safe_I_1(self, N,M,s):
        corrective_factor = math.lgamma(N+1) - np.sum([math.lgamma(i) for i in np.add(s, 1)])
        return corrective_factor + math.lgamma(N+M) - math.lgamma(M) - np.sum([math.lgamma(i) for i in np.add(s, 1)]) + 0.5*np.log((M-1)*math.pi) - 0.4 #here add in the corrective factor for adding an ordering m

    def logstar(self, n):
            if n < 2:
                return 1
            # return self.head(n) + 1
            else:
                return np.log(n) + np.log(np.log(n))
            
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



if __name__ == '__main__':

    def load(filename):
        df = pd.read_csv(filename)
        df = df[['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_SCORE']].dropna().reset_index()
        df_smiles = df['PUBCHEM_EXT_DATASOURCE_SMILES']
        df_activities = df['PUBCHEM_ACTIVITY_SCORE']
        return df_smiles, df_activities
    def load_sdf_files(filename):
        suppl = Chem.SDMolSupplier(filename)
        mols = []
        for mol in suppl:
            mols.append(Chem.MolToSmiles(mol))
        with open(filename,'rb') as f:
            txt = f.readlines()
            IC50s = []
            for idx, line in enumerate(txt):
                if '<pIC50>' in line.decode('utf-8'):
                    IC50s.append(float(txt[idx+1]))
        if len(IC50s) != len(mols):
            print(f'error for {filename}: {len(IC50s)=}, {len(mols)=}')
        else:
            return file, mols, IC50s # datasets.append([file, mols, IC50s])

    def get_symbols_rdkit(smiles):
        atoms_chars = []
        for smile in smiles:
            for at in Chem.MolFromSmiles(smile).GetAtoms():
                atom=Chem.RWMol() #from stackoverflow: 
                atom.AddAtom(at)
                atoms_chars.append(Chem.MolToSmiles(atom))
        return set(atoms_chars)
    
    def splitter(smile, syms):
        mol = []
        idx = 0
        idx_mx_g = max([i for i in map(len, syms)])
        while idx < len(smile):
            #print(idx)
            idx_mx = idx_mx_g
            for idx_mx in range(idx_mx, 0, -1): 
                if idx_mx > 1 and smile[idx:idx+idx_mx] in els and idx+idx_mx <= len(smile):
                    #print('first if')
                    mol.append(smile[idx:idx+idx_mx])
                    idx += idx_mx
                    break
                elif idx_mx == 1:
                    mol.append(smile[idx])
                    idx += 1
        return mol
    
    def p_counter():
        split_d = o.split_data(o.encoded_data)
        #split_d = o.encoded_data
        p.max_len = o.max_len #maybe set this for all relevant variable before?
        p.chars = o.chars
        p.rev_sym_dic = o.rev_sym_dic
        p.r_rule_dic = o.r_rule_dic
        p.rule_dic = o.rule_dic
        #split_d = list(zip(split_d, [p.max_len for i in range(len(split_d))]))
        #print(split_d)
        with Pool(processes=n_workers) as pool:
            dics = pool.imap_unordered(p.remove_invalid_subs, pool.imap_unordered(p.count_words, split_d))
            #print(dics) 
            dics = list(dics) #two
            #print(dics)
        return {k: sum(t.get(k, 0) for t in dics) for k in set.union(*[set(t) for t in dics])}

    def p_score_vals(cand_dic):
        p.org_codelens = o.org_codelens
        p.rule_dic = o.rule_dic
        p.r_rule_dic = o.r_rule_dic
        p.chars = o.chars
        initial_counts = {i:j for i,j in cand_dic.items() if len(i) == 1}
        #print('here' ,sum(initial_counts.values()))
        dics = o.split_data(list(cand_dic.items()))
        inp = [(dic, initial_counts) for dic in dics]
        with Pool(processes=n_workers) as pool:
            outs = pool.imap_unordered(p.score_subs,inp) #RETURNS LIST NEED TO COMBINE
            outs = list(outs)
        sub_score_dics = list(map(lambda x: x[2], outs)) #score dictionary
        sub_scores = {k: sum(t.get(k, 0) for t in sub_score_dics) for k in set.union(*[set(t) for t in sub_score_dics])} #combine dictionaries
        #print([(p.decode_sentence([list(i)]), j) for i,j in sub_scores.items()])
        if len(sub_scores) == 0: #if there are no substructures to score, return None
            return None
        as_dics = list(map(lambda x: x[1], outs))
        as_dic = {k: sum(t.get(k, 0) for t in as_dics) for k in set.union(*[set(t) for t in as_dics])} #combine dictionaries
        #print(as_dic)
        bst = min(sub_scores.items(), key=lambda x:x[1])
        #print(bst)
        #get current assertion total
        p.total = as_dic[bst[0]]  #here need to ensure that the previous assertion cost accounts for logstar(total length)
        #print(p.total)
        return bst

    def p_encode_words(max_seq,l):
        pieces = o.split_data(o.encoded_data)
        #pieces = o.encoded_data
        o.rule_dic[max_seq] = o.max_int+1 
        o.r_rule_dic[o.max_int+1] = max_seq
        unrolled_pair = o.unroll_pair(max_seq)
        o.unrolled_rule_dic[unrolled_pair] = o.max_int+1 
        o.r_unrolled_rule_dic[o.max_int+1] = unrolled_pair
        #THEN CHANGE P VARIABLE
        p.max_int = o.max_int
        inp = [(piece, max_seq, l) for piece in pieces]
        with Pool(processes=n_workers) as pool:
            res = pool.imap_unordered(p.encode_words, inp)
            res = list(res)
        encoded_data = sum(res, [])
        o.max_int += 1
        o.encoded_data = encoded_data #update class variable
        return max_seq, o.codelen

    def p_encode_step():
        s = pf()
        cand_dic = p_counter()
        e = pf()
        print(f'counting took {e-s}')
        s = pf()
        max_seq, o.codelen = p_score_vals(cand_dic)
        e = pf()
        print(f'scoring took {e-s}')
        s = pf()
        seq, codelen = p_encode_words(max_seq,len(max_seq))
        e = pf()
        print(f'encoding took {e-s}')
        #print(o.decode_sentence([list(seq)])[:-1], codelen)
        return seq, codelen

    def p_prune_dic():
        split_d = o.split_data(o.encoded_data)
        #split_d = o.encoded_data
        with Pool(processes=n_workers) as pool:
            res = pool.imap_unordered(p.get_used_tokens, split_d)
            res = list(res)
        used_tokens = set.union(*res)
        t_dic = dict()
        for token in o.rule_dic.values():
            if token > len(o.chars) and token in used_tokens:
                #add the rule for this token to the rule dictionary
                t_dic[token] = o.r_rule_dic[token]
        o.pruned_dic =  {j:i for i,j in t_dic.items()}
    
    def greedy_search(fname, MML=[], i=0):
        if len(MML) > 0:
            curr = MML[-1]
        else:
            curr = o.codelength()
            MML = [curr]
        curr = MML[-1]
        print(curr)
        st = pf()
        s, nxt = p_encode_step()
        print(nxt)
        p_prune_dic()
        ed = pf()
        #if this step was beneficial, add it to dic
        while nxt <= curr:
            i += 1
            print(i, o.decode_sentence([list(s)]), nxt, ed-st)
            #/home/bars2/MML87_Results
            #MML87_Results
            with open(fname, 'wb') as f:
                pickle.dump([MML, [o.decode_sentence([list(i)])[:-1] for i in o.pruned_dic], o], f)
            curr = nxt
            MML.append(curr)
            st = pf()
            s,nxt = p_encode_step()
            p_prune_dic()
            ed = pf()
        return MML, [o.decode_sentence([list(i)])[:-1] for i in o.pruned_dic], o

    def smi_tokenizer(smi):
        """
        Tokenize a SMILES molecule or reaction. Directly taken from Molecular Transformer paper.
        """
        pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return tokens
    
    

    with open('CHEMBL_36', 'rb') as f:
        smiles = pickle.load(f)
    smiles = [smi_tokenizer(s) for s in smiles]
    n_workers = multiprocessing.cpu_count()
    print(f'Running on {n_workers} core CPU')
    o = MML_Tokenizer(smiles, max_len=8, syntax_check=True, cores=n_workers)
    p = Parallel_Functions()
    p.total = None
    p.num_chars = o.num_chars
    greedy_search('CHEMBL_36_output_8')
    
    


















