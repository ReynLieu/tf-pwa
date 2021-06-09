import os
import numpy as np
import uproot
from tf_pwa.applications import gen_data, gen_mc
from tf_pwa.angle import LorentzVector as lv
from dat_cfit import get_var_from_data_dic
from fit import *


class DatWeight:
    def __init__(self, file_name, branch):
        with uproot.open(file_name) as f:
            self.eff_bin = f.get(branch)
        self.x_bins, self.y_bins = self.eff_bin.bins
        self.values = self.eff_bin.values

    def index_bin(self, x, xi):
        xi1 = np.expand_dims(xi, axis=-1)
        mask = (x[:, 0] < xi1) & (x[:, 1] > xi1)
        idx1, idx2 = np.nonzero(mask)
        idx = np.zeros_like(xi, dtype="int64")
        idx[idx1] = idx2
        return idx

    def weight(self, m2_13, m2_23, eff_0_correct=False):
        x_idx = self.index_bin(self.x_bins, m2_13)
        y_idx = self.index_bin(self.y_bins, m2_23)
        values = self.values[x_idx, y_idx]
        if eff_0_correct:
            values = self._where_eff_0_(values, x_idx, y_idx)
        return values

    def _where_eff_0_(self, values, x_idx, y_idx):
        new_vals = values.copy()
        for tr_fl, idx in zip(values > 0, range(len(values))):
            if not tr_fl:        
                tmp = 0
                n = 0
                tmp_val = 0
                for i in range(-1,2):
                    for j in range(-1,2):
                        tmp_val = self.values[x_idx[idx]+i, y_idx[idx]+j]
                        if tmp_val > 0:
                            tmp += tmp_val
                            n += 1
                if n == 0:
                    new_vals[idx] = 0
                else:
                    new_vals[idx] = tmp/n
        return new_vals

Bz = 5.27963
Dz = 1.86483
Dsp = 1.96834
pim = 0.13957039
Bp = 5.27934
Dm = 1.86965
pip = pim

def gen_mc_from_eff_map(mcfile, Nmc, eff_file):
    if mcfile[-9:-7] == "Bz":
        pf = gen_mc(Bz, [Dz, Dsp, pim], Nmc)
    elif mcfile[-9:-7] == "Bp":
        pf = gen_mc(Bp, [Dm, Dsp, pip], Nmc)
    else:
        raise
    p1, p2, p3 = pf.reshape((-1, 3, 4)).transpose((1, 0, 2))
    m13sq = lv.M2(p1 + p3)
    m23sq = lv.M2(p2 + p3)

    eff = DatWeight(eff_file, "RegDalitzEfficiency")
    maxeffval = np.max(eff.values)
    uni_rdm = np.random.uniform(0, maxeffval, Nmc)
    weff = eff.weight(m13sq, m23sq, eff_0_correct=True)
    if mcfile[-9:-7] == "Bz":
        weff[m13sq<2.05**2] = 0 # some will become mistakenly nonzero after eff_0_correct
    pf = pf.reshape(-1, 3, 4)
    mc_four_momentum = pf[weff>uni_rdm]
    Nmcsample = len(mc_four_momentum)
    mc_four_momentum = mc_four_momentum.reshape(-1, 4)
    np.savetxt(mcfile, mc_four_momentum)

def gen_toyBz(amp):
    ndata = 633; nbg = 480; wbg = 0.131039;
    mcfile = "toystudy/toy/BzMC1.dat"
    genfile = "toystudy/toy/BzData1.dat"
    bgfile = "../dataDDspi/dat/B0toD0Dspi_BSB_Run1.dat"
    efffile = "../dataDDspi/B0toD0Dspi/EffMapCorr/EffMap_B0toD0Dspi_Run1.root"
    gen_mc_from_eff_map(mcfile, ndata*10, efffile)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)

    ndata = 2753; nbg = 2047; wbg = 0.103967;
    mcfile = "toystudy/toy/BzMC2.dat"
    genfile = "toystudy/toy/BzData2.dat"
    bgfile = "../dataDDspi/dat/B0toD0Dspi_BSB_Run2.dat"
    efffile = "../dataDDspi/B0toD0Dspi/EffMapCorr/EffMap_B0toD0Dspi_Run2.root"
    gen_mc_from_eff_map(mcfile, ndata*10, efffile)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)

    ndata = 199; nbg = 133; wbg = 0.143792;
    mcfile = "toystudy/toy/BzMC3.dat"
    genfile = "toystudy/toy/BzData3.dat"
    bgfile = "../dataDDspi/dat/B0toD0_K3piDspi_BSB_Run1.dat"
    efffile = "../dataDDspi/B0toD0_K3piDspi/EffMapCorr/EffMap_B0toD0_K3piDspi_Run1.root"
    gen_mc_from_eff_map(mcfile, ndata*10, efffile)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)

    ndata = 835; nbg = 969; wbg = 0.106173;
    mcfile = "toystudy/toy/BzMC4.dat"
    genfile = "toystudy/toy/BzData4.dat"
    bgfile = "../dataDDspi/dat/B0toD0_K3piDspi_BSB_Run2.dat"
    efffile = "../dataDDspi/B0toD0_K3piDspi/EffMapCorr/EffMap_B0toD0_K3piDspi_Run2.root"
    gen_mc_from_eff_map(mcfile, ndata*10, efffile)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)

def gen_toyBp(amp):
    ndata = 797; nbg = 407; wbg = 0.0861442;
    mcfile = "toystudy/toy/BpMC1.dat"
    genfile = "toystudy/toy/BpData1.dat"
    bgfile = "../dataDDspi/dat/BptoDmDspi_BSB_Run1.dat"
    efffile = "../dataDDspi/BptoDmDspi/EffMapCorr/EffMap_BptoDmDspi_Run1.root"
    gen_mc_from_eff_map(mcfile, ndata*10, efffile)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)

    ndata = 3143; nbg = 2471; wbg = 0.0554937;
    mcfile = "toystudy/toy/BpMC2.dat"
    genfile = "toystudy/toy/BpData2.dat"
    bgfile = "../dataDDspi/dat/BptoDmDspi_BSB_Run2.dat"
    efffile = "../dataDDspi/BptoDmDspi/EffMapCorr/EffMap_BptoDmDspi_Run2.root"
    gen_mc_from_eff_map(mcfile, ndata*10, efffile)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)

def gen_weigths_for_cfit(mode, config):
    if mode == "Bz":
        channels = ["B0toD0Dspi_Run1", "B0toD0Dspi_Run2", "B0toD0_K3piDspi_Run1", "B0toD0_K3piDspi_Run2"]
    elif mode == "Bp":
        channels = ["BptoDmDspi_Run1", "BptoDmDspi_Run2"]
    datas = config.get_data("data")
    phsps = config.get_data("phsp")
    for i in range(len(datas)):
        ch = channels[i]
        md = ch[:-5]
        bkg = DatWeight(f"../dataDDspi/{md}/SidebandMap/BSBMap_{ch}_Smooth.root", "Hist_pos")
        eff = DatWeight(f"../dataDDspi/{md}/EffMapCorr/EffMap_{ch}.root", "RegDalitzEfficiency")
        # calculate bkg_value, eff_value for data
        Dvars, SqDvars = get_var_from_data_dic(datas[i], md)
        save_file = f"toystudy/toy/{mode}Data{i+1}_value.dat"
        wbkg = bkg.weight(*SqDvars)
        weff = eff.weight(*Dvars, eff_0_correct=True)
        np.savetxt(save_file, wbkg/weff)
        # calculate bkg_value, eff_value for phsp MC
        Dvars, SqDvars = get_var_from_data_dic(phsps[i], md)
        save_file = f"toystudy/toy/{mode}MC{i+1}_value.dat"
        wbkg = bkg.weight(*SqDvars)
        weff = eff.weight(*Dvars, eff_0_correct=True)
        np.savetxt(save_file, wbkg/weff)


def fit_null(config):
    #config.get_amplitudes()[0].vm.rp2xy_all()
    fit_result = config.fit(batch=150000, method="BFGS")
    return fit_result

def fit_Z(config, params, ZJ=0, loop=1):
    config.set_params(params)
    config.set_params({f"B->Z{ZJ}.DZ{ZJ}->Ds.Pi_total_0r": 0})
    frt = config.fit(batch=150000, method="BFGS")
    min_nll = frt.min_nll
    for i in range(loop-1):
        config.reinit_params()
        config.set_params(params)
        config.set_params({f"Z{ZJ}_mass": 2.9})
        config.set_params({f"Z{ZJ}_width": 0.15})
        config.set_params({f"B->Z{ZJ}.DZ{ZJ}->Ds.Pi_total_0r": 0})
        frt = config.fit(batch=150000, method="BFGS")
        if min_nll - frt.min_nll > 1e-6:
            print("$$$New Min found")
            min_nll = frt.min_nll
            frt = frt
    return frt

def sig_test(sfit=True, cfit=True, null="", alternative="Z0", param_file=None, fitloop=1):
    if param_file is None:
        param_file = f"toystudy/params/base{null}_s.json"
    Sconfig = MultiConfig([f"toystudy/StoyBz{null}.yml", f"toystudy/StoyBp{null}.yml"], total_same=True)
    Sconfig.set_params(param_file)
    ampBz, ampBp = Sconfig.get_amplitudes()
    gen_toyBz(ampBz)
    gen_toyBp(ampBp)
    Sdnll = 0
    Cdnll = 0
    if sfit:
        print("### Null Sfit")
        Sfr0 = fit_null(Sconfig)
        print("### Alternative Sfit")
        SconfigZ = MultiConfig([f"toystudy/StoyBz{alternative}.yml", f"toystudy/StoyBp{alternative}.yml"], total_same=True)
        if null == "" or null == "Z1":
            SfrZ = fit_Z(SconfigZ, Sfr0.params, ZJ=0, loop=fitloop)
        elif null == "Z0":
            SfrZ = fit_Z(SconfigZ, Sfr0.params, ZJ=1, loop=fitloop)
        Sdnll = Sfr0.min_nll - SfrZ.min_nll

    if cfit:
        print("### Null Cfit")
        Cconfig = MultiConfig([f"toystudy/CtoyBz{null}.yml", f"toystudy/CtoyBp{null}.yml"], total_same=True)
        gen_weigths_for_cfit("Bz", Sconfig.configs[0]) # only for access to data and phsp
        gen_weigths_for_cfit("Bp", Sconfig.configs[1])
        Cconfig.set_params(param_file)
        Cfr0 = fit_null(Cconfig)
        print("### Alternative Cfit")
        CconfigZ = MultiConfig([f"toystudy/CtoyBz{alternative}.yml", f"toystudy/CtoyBp{alternative}.yml"], total_same=True)
        if null == "" or null == "Z1":
            CfrZ = fit_Z(CconfigZ, Cfr0.params, ZJ=0, loop=fitloop)
        elif null == "Z0":
            CfrZ = fit_Z(CconfigZ, Cfr0.params, ZJ=1, loop=fitloop)
        Cdnll = Cfr0.min_nll - CfrZ.min_nll

    return Sdnll, Cdnll


def main(Ntoy):
    SdNLL = []
    CdNLL = []
    for i in range(Ntoy):
        print("##### Start toy {}".format(i))
        Sdnll, Cdnll = sig_test(sfit=True, cfit=True, null="Z0", alternative="Z01", param_file="toystudy/params/baseZ0_c.json") # edit
        print("$$$$$dnll:", Sdnll, Cdnll)
        SdNLL.append(Sdnll)
        CdNLL.append(Cdnll)
        print(f"deltaNLL for Sfit:\n{SdNLL}")
        print(f"deltaNLL for Cfit:\n{CdNLL}")
    return SdNLL, CdNLL

if __name__ == "__main__":
    if not os.path.exists("toystudy/toy"):
        os.mkdir("toystudy/toy")
    SdNLL, CdNLL = main(100) # usually it goes OOM before reach the 100 times
    print(f"########## deltaNLL for Sfit:\n{SdNLL}")
    print(f"########## deltaNLL for Cfit:\n{CdNLL}")

