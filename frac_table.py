import re
import numpy as np

frac_txt = """
D2_2460 0.252 +/- 0.007
D1_2600 0.044 +/- 0.004
D1_2600xD2_2460 0.00087 +/- 0.00009
D1_2010 0.124 +/- 0.011
D1_2010xD1_2600 -0.0028 +/- 0.0030
D1_2010xD2_2460 -0.00065 +/- 0.00010
DPi_S 0.584 +/- 0.010
DPi_SxD1_2010 -0.00146 +/- 0.00018
DPi_SxD1_2600 -0.00035 +/- 0.00006
DPi_SxD2_2460 0.00023 +/- 0.00006
"""

def get_point(s):
    partten = re.compile(r"([^\s]+)\s+([+-.e1234567890]+)\s+")
    ret = {}
    for i in s.split("\n"):
        g = partten.match(i)
        #print(g)
        if g:
            name = g.group(1).split("x")
            #print(name)
            frac = float(g.group(2))
            if len(name) == 1:
                l, r = name*2
            elif len(name) == 2:
                l, r = name
            else:
                raise Exception("error {}".format(name))
            if l not in ret:
                ret[l] = {}
            ret[l][r] = frac
    return ret


def get_table(s):
    #print(s)
    idx = list(s)
    n_idx = len(idx)
    idx_map = dict(zip(idx, range(n_idx)))
    #print(idx_map)
    ret = []
    for i in range(n_idx):
        ret.append([0.0 for j in range(n_idx)])
    for i, k in s.items():
        for j, v in k.items():
            ret[idx_map[i]][idx_map[j]] = v
    return idx, ret


def frac_table(frac_txt):
    s = get_point(frac_txt)
    idx, table = get_table(s)
    import pprint
    #pprint.pprint(idx)
    #print(idx)
    #print(table)

    # remove D*
    #idx = idx[1:]
    #table = np.array(table)[1:,1:]

    ret = []
    for i, k in enumerate(table):
        tmp = []
        for j, v in enumerate(k):
            if i < j:
                tmp.append("-")
            else:
                tmp.append("{:.3f}".format(v))
        ret.append(tmp)
    #pprint.pprint(ret)
    for i, k in zip(idx, ret):
        print(i, end="\t")
        for v in k:
            print(v, end="\t")
        print()
    print("Total sum:", np.sum(table))
    print("Non-interference sum:", np.sum(np.diagonal(table)))


if __name__=="__main__":
    frac_table(frac_txt)
            
    
