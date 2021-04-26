##### this script is used to generate MC according to efficiency maps
##### put this script under ../dataDstarDK
import sys
import os.path

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_dir + '/..')

import uproot
import numpy as np
from tf_pwa.phasespace import PhaseSpaceGenerator
from tf_pwa.angle import Vector3 as v3, LorentzVector as lv
import matplotlib.pyplot as plt


M_pipm = 0.13957061
#M_pi0 = 0.1349766
M_Kpm = 0.493677
#M_Dpm = 1.86961
M_D0 = 1.86483
M_Dstarpm = 2.01026
#M_Bpm = 5.27926
M_B0 = 5.27963

def generate_mc(num):
    a = PhaseSpaceGenerator(M_B0, [M_Dstarpm, M_D0, M_Kpm])
    b = PhaseSpaceGenerator(M_Dstarpm, [M_D0, M_pipm])
    p_D1, p_D2, p_K = a.generate(num)
    p_D, p_pi = b.generate(num)
    return [i.numpy() for i in [p_D1, p_D2, p_K, p_D, p_pi]]


class EffWeight:
    def __init__(self, root_file):
        self.f = uproot.open(root_file)
        self.eff_bin = [self.f.get("RegDalitzEfficiency_bin{}".format(i)) for i in range(5)]
        self.x_bins, self.y_bins = self.eff_bin[0].bins  # assert all bins same
        self.values = np.array([i.values for i in self.eff_bin])

    def index_bin(self, x, xi):
        xi1 = np.expand_dims(xi, axis=-1)
        mask = (x[:, 0] < xi1) & (x[:, 1] > xi1)
        idx1, idx2 = np.nonzero(mask)
        idx = np.zeros_like(xi, dtype="int64")
        idx[idx1] = idx2
        return idx

    def eff_weight(self, cos_hel, m2_13, m2_23):
        n_histo = (3 - np.floor((cos_hel + 1) / 0.5) + 1).astype("int64")  # [-1,1]->[4,3,2,1]
        x_idx = self.index_bin(self.x_bins, m2_13)
        y_idx = self.index_bin(self.y_bins, m2_23)
        values = self.values[n_histo, x_idx, y_idx]
        values = self._where_eff_0_(values, n_histo, x_idx, y_idx)
        return values

    def _where_eff_0_(self, values, n_histo, x_idx, y_idx):
        new_vals = values.copy()
        for tr_fl, idx in zip(values > 0, range(len(n_histo))):
            if not tr_fl:        
                tmp = 0
                n = 0
                tmp_val = 0
                for i in range(-1,2):
                    for j in range(-1,2):
                        tmp_val = self.values[n_histo[idx], x_idx[idx]+i, y_idx[idx]+j]
                        if tmp_val > 0:
                            tmp += tmp_val
                            n += 1
                new_vals[idx] = tmp/n
        return new_vals

def _generate_mc_eff(num, eff_file):
    p_D1, p_D2, p_K, p_D, p_pi = generate_mc(num)
    m2_13 = lv.M2(p_D1 + p_K)
    m2_23 = lv.M2(p_D2 + p_K)
    cos_theta = v3.cos_theta(lv.vect(p_D1), lv.vect(p_D))
    #plt.hist(cos_theta, bins=100)
    #plt.savefig("cos_theta.png")
    eff = EffWeight(eff_file)
    weight = eff.eff_weight(cos_theta, m2_13, m2_23)
    rnd = np.random.random(num)
    mask = weight > rnd
    p_D1, p_D2, p_K, p_D, p_pi = [i[mask] for i in [p_D1, p_D2, p_K, p_D, p_pi]]

    #m2_13 = lv.M2(p_D1 + p_K)
    #m2_23 = lv.M2(p_D2 + p_K)
    #cos_theta = v3.cos_theta(lv.vect(p_D1), lv.vect(p_D))
    #plt.hist(cos_theta, bins=100)
    #plt.savefig("cos_theta.png")
    #cos_theta = v3.cos_theta(lv.vect(p_D1), np.array([[0.,0.,1.]]*p_D1.__len__()))
    #plt.clf()
    #plt.hist(cos_theta, bins=100)
    #plt.savefig("cos_Dstar.png")
    return p_D1, p_D2, p_K, p_D, p_pi


def gen_eff_mc(num, rn, channel):
    print("##########", channel, rn)
    if rn == 1:
        eff_file = "eff/Efficiency_"+channel+"_run1.root"
    elif rn == 2:
        eff_file = "eff/Efficiency_"+channel+"_run2.root"
    p_D1, p_D2, p_K, p_D, p_pi = _generate_mc_eff(num, eff_file)
    p_D, p_pi = [lv.rest_vector(lv.neg(p_D1), i) for i in [p_D, p_pi]]  # boost to B
    print("Number of events: ", np.shape(p_D)[0])
    data = np.array([p_D2, p_K, p_D, p_pi])
    data = np.transpose(data, (1, 0, 2))
    
    np.savetxt("PHSP_"+channel+"Run"+str(rn)+".dat", data.reshape(-1, 4))
    #bins, x, y, _ = plt.hist2d(lv.M2(p_D1 + p_K), lv.M2(p_D2 + p_K), bins=400)
    #plt.clf()
    #plt.imshow(bins,origin='lower',extent=[x[0],x[-1],y[0],y[-1]])
    #plt.contourf(*np.meshgrid(x[1:]/2+x[:-1]/2, y[1:]/2+y[:-1]/2), bins.astype("float"))
    #plt.xlabel("$m^2_{D^{*-}K^+}(GeV^2)$")
    #plt.ylabel("$m^2_{D^{0}K^+}(GeV^2)$")
    #plt.colorbar()
    #plt.savefig("m13_m23"+channel+str(rn))

from tf_pwa.applications import gen_mc
def gen_PHSP_mc(Nmc):
    data3b = gen_mc(M_B0, [M_Dstarpm, M_D0, M_Kpm], Nmc)
    p_D1 = data3b[0::3]
    p_D2 = data3b[1::3]
    p_K = data3b[2::3]
    dataDst = gen_mc(M_Dstarpm, [M_D0, M_pipm], Nmc)
    p_D = dataDst[0::2]
    p_pi = dataDst[1::2]
    p_D, p_pi = [lv.rest_vector(lv.neg(p_D1), i) for i in [p_D, p_pi]]  # boost to B
    data = np.array([p_D2, p_K, p_D, p_pi])
    data = np.transpose(data, (1, 0, 2))
    np.savetxt("PHSP30w.dat", data.reshape(-1, 4))
    #print(data)

if __name__ == "__main__":
    gen_eff_mc(40000, rn=1, channel="DstDK")
    gen_eff_mc(150000, rn=2, channel="DstDK")
    gen_eff_mc(70000, rn=1, channel="Dst_K3pi")
    gen_eff_mc(250000, rn=2, channel="Dst_K3pi")
    gen_eff_mc(70000, rn=1, channel="D_K3pi")
    gen_eff_mc(250000, rn=2, channel="D_K3pi")
    #gen_PHSP_mc(300000)
