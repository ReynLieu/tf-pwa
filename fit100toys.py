#!/usr/bin/env python3

# avoid using Xwindow
import matplotlib

matplotlib.use("agg")

from tf_pwa.config_loader import ConfigLoader, MultiConfig
from pprint import pprint
from tf_pwa.utils import error_print
import tensorflow as tf
import json
import time
import uproot
import numpy as np
import os


# examples of custom particle model
from tf_pwa.amp import simple_resonance
from tf_pwa.experimental import extra_amp, extra_data
from tf_pwa.applications import gen_data, gen_mc
from tf_pwa.phasespace import PhaseSpaceGenerator
from tf_pwa.angle import Vector3 as v3, LorentzVector as lv
from tf_pwa.variable import VarsManager
from fit import *

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
                if n is 0:
                    print("$#@!!check", n_histo[idx], x_idx[idx], y_idx[idx])
                    new_vals[idx] = 0
                else:
                    new_vals[idx] = tmp/n
        return new_vals

def _generate_mc_eff(num, eff_file):
    p_D1, p_D2, p_K, p_D, p_pi = generate_mc(num)
    m2_13 = lv.M2(p_D1 + p_K)
    m2_23 = lv.M2(p_D2 + p_K)
    cos_theta = v3.cos_theta(lv.vect(p_D1), lv.vect(p_D))
    eff = EffWeight(eff_file)
    weight = eff.eff_weight(cos_theta, m2_13, m2_23)
    rnd = np.random.random(num)
    mask = weight > rnd
    p_D1, p_D2, p_K, p_D, p_pi = [i[mask] for i in [p_D1, p_D2, p_K, p_D, p_pi]]
    return p_D1, p_D2, p_K, p_D, p_pi

def gen_eff_mc(num, rn, channel):
    if rn == 1:
        eff_file = "../dataDstarDK/eff/Efficiency_"+channel+"_run1.root"
    elif rn == 2:
        eff_file = "../dataDstarDK/eff/Efficiency_"+channel+"_run2.root"
    p_D1, p_D2, p_K, p_D, p_pi = _generate_mc_eff(num, eff_file)
    p_D, p_pi = [lv.rest_vector(lv.neg(p_D1), i) for i in [p_D, p_pi]]  # boost to B
    print("Number of events: ", np.shape(p_D)[0])
    data = np.array([p_D2, p_K, p_D, p_pi])
    data = np.transpose(data, (1, 0, 2))
    data = data.reshape(-1, 4)
    np.savetxt("toy/toydat/toyMC_"+channel+"Run"+str(rn)+".dat", data)
    return data


def gen_toy(amp):
    channel = "DstDK"; rn = 1
    print("##########", channel, rn)
    ndata = 618; nbg = 53; wbg = 0.571; nmc = 40000
    gen_eff_mc(nmc//2, rn, channel)
    gen_data(amp, ndata, mcfile="toy/toydat/toyMC_"+channel+"Run"+str(rn)+".dat", 
            genfile="toy/toydat/toy_"+channel+"Run"+str(rn)+".dat", Poisson_fluc=True, 
            Nbg=nbg, wbg=wbg, bgfile="../dataDstarDK/dat/sb_"+channel+"Run"+str(rn)+".dat")
    gen_eff_mc(nmc, rn, channel)

    rn = 2
    print("##########", channel, rn)
    ndata = 2494; nbg = 213; wbg = 0.585; nmc = 150000
    gen_eff_mc(nmc//2, rn, channel)
    gen_data(amp, ndata, mcfile="toy/toydat/toyMC_"+channel+"Run"+str(rn)+".dat", 
            genfile="toy/toydat/toy_"+channel+"Run"+str(rn)+".dat", Poisson_fluc=True, 
            Nbg=nbg, wbg=wbg, bgfile="../dataDstarDK/dat/sb_"+channel+"Run"+str(rn)+".dat")
    gen_eff_mc(nmc, rn, channel)
    
    channel = "Dst_K3pi"; rn = 1
    print("##########", channel, rn)
    ndata = 332; nbg = 29; wbg = 0.590; nmc = 70000
    gen_eff_mc(nmc//2, rn, channel)
    gen_data(amp, ndata, mcfile="toy/toydat/toyMC_"+channel+"Run"+str(rn)+".dat", 
            genfile="toy/toydat/toy_"+channel+"Run"+str(rn)+".dat", Poisson_fluc=True, 
            Nbg=nbg, wbg=wbg, bgfile="../dataDstarDK/dat/sb_"+channel+"Run"+str(rn)+".dat")
    gen_eff_mc(nmc, rn, channel)

    rn = 2
    print("##########", channel, rn)
    ndata = 1757; nbg = 153; wbg = 0.587; nmc = 250000
    gen_eff_mc(nmc//2, rn, channel)
    gen_data(amp, ndata, mcfile="toy/toydat/toyMC_"+channel+"Run"+str(rn)+".dat", 
            genfile="toy/toydat/toy_"+channel+"Run"+str(rn)+".dat", Poisson_fluc=True, 
            Nbg=nbg, wbg=wbg, bgfile="../dataDstarDK/dat/sb_"+channel+"Run"+str(rn)+".dat")
    gen_eff_mc(nmc, rn, channel)
    
    channel = "D_K3pi"; rn = 1
    print("##########", channel, rn)
    ndata = 341; nbg = 29; wbg = 0.578; nmc = 70000
    gen_eff_mc(nmc//2, rn, channel)
    gen_data(amp, ndata, mcfile="toy/toydat/toyMC_"+channel+"Run"+str(rn)+".dat", 
            genfile="toy/toydat/toy_"+channel+"Run"+str(rn)+".dat", Poisson_fluc=True, 
            Nbg=nbg, wbg=wbg, bgfile="../dataDstarDK/dat/sb_"+channel+"Run"+str(rn)+".dat")
    gen_eff_mc(nmc, rn, channel)

    rn = 2
    print("##########", channel, rn)
    ndata = 1586; nbg = 101; wbg = 0.591; nmc = 250000
    gen_eff_mc(nmc//2, rn, channel)
    gen_data(amp, ndata, mcfile="toy/toydat/toyMC_"+channel+"Run"+str(rn)+".dat", 
            genfile="toy/toydat/toy_"+channel+"Run"+str(rn)+".dat", Poisson_fluc=True, 
            Nbg=nbg, wbg=wbg, bgfile="../dataDstarDK/dat/sb_"+channel+"Run"+str(rn)+".dat")
    gen_eff_mc(nmc, rn, channel)


fp = {}
fe = {}
fi = {'Ds2_2573p_mass': 2.5691, 'Ds2_2573p_width': 0.0169, 'Ds3_2860p_mass': 2.8605, 'Ds3_2860p_width': 0.053, 'Ds1_2700p_mass': 2.7083, 'Ds1_2700p_width': 0.12, 'Ds1_2860p_mass': 2.859, 'Ds1_2860p_width': 0.159, 'NR(DK)PS_a': -0.535567448777523, 'A->Ds2_2573p.BDs2_2573p->C.DB->E.F_total_0r': 1.0, 'A->Ds2_2573p.BDs2_2573p->C.DB->E.F_total_0i': 0.0, 'A->Ds2_2573p.B_g_ls_0r': 1.0, 'A->Ds2_2573p.B_g_ls_0i': 0.0, 'A->Ds2_2573p.B_g_ls_1r': 0.4291058731711176, 'A->Ds2_2573p.B_g_ls_1i': -0.24168507377186468, 'A->Ds2_2573p.B_g_ls_2r': 0.208784062722235, 'A->Ds2_2573p.B_g_ls_2i': 0.5106013264135965, 'Ds2_2573p->C.D_g_ls_0r': 1.0, 'Ds2_2573p->C.D_g_ls_0i': 0.0, 'B->E.F_g_ls_0r': 1.0, 'B->E.F_g_ls_0i': 0.0, 'A->Ds3_2860p.BDs3_2860p->C.DB->E.F_total_0r': 0.10965221622751667, 'A->Ds3_2860p.BDs3_2860p->C.DB->E.F_total_0i': 0.6984742184569209, 'A->Ds3_2860p.B_g_ls_0r': 1.0, 'A->Ds3_2860p.B_g_ls_0i': 0.0, 'A->Ds3_2860p.B_g_ls_1r': 1.3527116708826925, 'A->Ds3_2860p.B_g_ls_1i': 1.1206346233802222, 'A->Ds3_2860p.B_g_ls_2r': 0.21573794186987014, 'A->Ds3_2860p.B_g_ls_2i': -0.3750589270083364, 'Ds3_2860p->C.D_g_ls_0r': 1.0, 'Ds3_2860p->C.D_g_ls_0i': 0.0, 'A->Ds1_2700p.BDs1_2700p->C.DB->E.F_total_0r': -9.470553799417193, 'A->Ds1_2700p.BDs1_2700p->C.DB->E.F_total_0i': -0.9069078772379191, 'A->Ds1_2700p.B_g_ls_0r': 1.0, 'A->Ds1_2700p.B_g_ls_0i': 0.0, 'A->Ds1_2700p.B_g_ls_1r': -0.3899977991126207, 'A->Ds1_2700p.B_g_ls_1i': 0.021096654366939294, 'A->Ds1_2700p.B_g_ls_2r': -0.2177468119960149, 'A->Ds1_2700p.B_g_ls_2i': 0.116700512588848, 'Ds1_2700p->C.D_g_ls_0r': 1.0, 'Ds1_2700p->C.D_g_ls_0i': 0.0, 'A->Ds1_2860p.BDs1_2860p->C.DB->E.F_total_0r': 2.4361554698302026, 'A->Ds1_2860p.BDs1_2860p->C.DB->E.F_total_0i': -2.4168496731125177, 'A->Ds1_2860p.B_g_ls_0r': 1.0, 'A->Ds1_2860p.B_g_ls_0i': 0.0, 'A->Ds1_2860p.B_g_ls_1r': 0.07830775405269035, 'A->Ds1_2860p.B_g_ls_1i': 0.4434625255708179, 'A->Ds1_2860p.B_g_ls_2r': -0.43270840520982723, 'A->Ds1_2860p.B_g_ls_2i': -0.3001433811439401, 'Ds1_2860p->C.D_g_ls_0r': 1.0, 'Ds1_2860p->C.D_g_ls_0i': 0.0, 'A->NR(DK)PS.BNR(DK)PS->C.DB->E.F_total_0r': -13.448490128757102, 'A->NR(DK)PS.BNR(DK)PS->C.DB->E.F_total_0i': 12.209728910426074, 'A->NR(DK)PS.B_g_ls_0r': 1.0, 'A->NR(DK)PS.B_g_ls_0i': 0.0, 'NR(DK)PS->C.D_g_ls_0r': 1.0, 'NR(DK)PS->C.D_g_ls_0i': 0.0, 'A->NR(DK)SP.BNR(DK)SP->C.DB->E.F_total_0r': -10.13942697422275, 'A->NR(DK)SP.BNR(DK)SP->C.DB->E.F_total_0i': -7.579433191699624, 'A->NR(DK)SP.B_g_ls_0r': 1.0, 'A->NR(DK)SP.B_g_ls_0i': 0.0, 'NR(DK)SP->C.D_g_ls_0r': 1.0, 'NR(DK)SP->C.D_g_ls_0i': 0.0, 'weight_injectMC': 0.001, 'weight_injectMC_1': 0.001, 'weight_injectMC_2': 0.001, 'weight_injectMC_3': 0.001, 'weight_injectMC_4': 0.001, 'weight_injectMC_5': 0.001}
def fit(i, n):
    bad_toy = False
    vm = VarsManager()
    config = ConfigLoader("toy/config_toy.yml", vm=vm)
    #config0 = ConfigLoader("toy/config_toy0.yml", vm=vm)
    config0 = config
    
    amp = config0.get_amplitude()
    config.set_params("toy/toy_params.json")
    if i==0:
        for v in amp.vm.trainable_vars:
            fp[v]=[]
            fe[v]=[]
    gen_toy(amp)
    #config.fit(batch=150000, method="BFGS")
    fit_result = config0.fit(batch=150000, method="BFGS")
    if not fit_result.success:
        print("$$$$$failed")
        return

    for ni in range(n-1):
        config.set_params("toy/toy_params.json")
        config.reinit_params()
        fit_res = config.fit(batch=150000, method="BFGS")
        if fit_res.success and (fit_res.min_nll < fit_result.min_nll):
            fit_result = fit_res
            if fit_result.min_nll - fit_res.min_nll > 1e-6:
                print("$$$ New Min Found")
    config.set_params(fit_result.params)

    amp.cached_fun = amp.decay_group.sum_amp # turn off use_tf_function
    amp.vm.rp2xy_all()
    fit_result = config0.fit(batch=150000, method="BFGS")
    if not fit_result.success:
        print("$$$$$failed")
        return
    fit_error = config0.get_params_error(fit_result, batch=30000)
    fit_result.set_error(fit_error)
    for v in amp.vm.trainable_vars:
        fp[v].append(fit_result.params[v])
        fe[v].append(fit_result.error[v])
        if v!='NR(DK)PS_a' and abs(fit_result.params[v] - fi[v])/fit_result.error[v] > 5:
            bad_toy = True
            print("!!!BAD_TOY", v)
        #print(v, fp[v], fe[v])
    print(f"@@@@@values\n{fp}")
    print(f"@@@@@errors\n{fe}")
    return bad_toy




def main(Ntoy):
    """entry point of fit. add some arguments in commond line"""
    if not os.path.exists("toy/toydat"):
        os.mkdir("toy/toydat")
    print("$$$$$Start fit100toys")
    for i in range(Ntoy):
        bad_toy = fit(i, n=10)
        if bad_toy:
            break

if __name__ == "__main__":
    main(40)
