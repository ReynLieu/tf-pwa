import os
import numpy as np
import uproot
from tf_pwa.applications import gen_data, gen_mc
from tf_pwa.angle import LorentzVector as lv
from tf_pwa.cal_angle import prepare_data_from_decay
from dat_cfit import DatWeight, get_var_from_data_dic, gen_bkg_sample
from fit import *

Bz = 5.27963
Dz = 1.86483
Dsp = 1.96834
pim = 0.13957039
Bp = 5.27934
Dm = 1.86965
pip = pim

def gen_mc_from_eff_map(mcfile, Nmc, eff_file):
    if mcfile.split('/')[-1][:2] == "Bz":
        pf = gen_mc(Bz, [Dz, Dsp, pim], Nmc)
    elif mcfile.split('/')[-1][:2] == "Bp":
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
    print(f"### Generate {Nmcsample} MC")
    mc_four_momentum = mc_four_momentum.reshape(-1, 4)
    np.savetxt(mcfile, mc_four_momentum)

def gen_toyBz(amp, config, gen_bkg_for_sfit=True):
    ndata = 633; nbg = 480; wbg = 0.131039;
    md = "B0toD0Dspi"; ch = "B0toD0Dspi_Run1";
    mcfile = "toystudy/toy/BzMC1.dat"
    genfile = "toystudy/toy/BzData1.dat"
    efffile = f"../dataDDspi/{md}/EffMapCorr/EffMap_{ch}.root"
    bkgpdf = DatWeight(f"../dataDDspi/{md}/SidebandMap/BSBMap_{ch}_Smooth.root", "Hist_pos")
    #bgfile = "../dataDDspi/dat/B0toD0Dspi_BSB_Run1.dat"
    bgfile = "toystudy/toy/BzBKG1.dat"
    gen_mc_from_eff_map(mcfile, ndata*100, efffile)
    if nbg*wbg != 0: # only nbg*wbg is used, in blending such amount of bkg into data
        gen_bkg_sample(bgfile, int(100*nbg*wbg), config, bkgpdf, "Bz", md)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)
    if gen_bkg_for_sfit and nbg*wbg != 0: # generate bkg sample again for bkg subtraction in sfit
        fourmom = gen_bkg_sample(bgfile, int(10*nbg), config, bkgpdf, "Bz", md)
        nsb = nbg # np.random.poisson(nbg)
        if len(fourmom) < 3*nsb:
            raise Exception("$$$ not enough bkg MC sample")
        np.savetxt(bgfile, fourmom[:3*nsb])

    ndata = 2753; nbg = 2047; wbg = 0.103967;
    md = "B0toD0Dspi"; ch = "B0toD0Dspi_Run2";
    mcfile = "toystudy/toy/BzMC2.dat"
    genfile = "toystudy/toy/BzData2.dat"
    efffile = f"../dataDDspi/{md}/EffMapCorr/EffMap_{ch}.root"
    bkgpdf = DatWeight(f"../dataDDspi/{md}/SidebandMap/BSBMap_{ch}_Smooth.root", "Hist_pos")
    #bgfile = "../dataDDspi/dat/B0toD0Dspi_BSB_Run2.dat"
    bgfile = "toystudy/toy/BzBKG2.dat"
    gen_mc_from_eff_map(mcfile, ndata*100, efffile)
    if nbg*wbg != 0:
        gen_bkg_sample(bgfile, int(100*nbg*wbg), config, bkgpdf, "Bz", md)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)
    if gen_bkg_for_sfit and nbg*wbg != 0:
        fourmom = gen_bkg_sample(bgfile, int(10*nbg), config, bkgpdf, "Bz", md)
        nsb = nbg # np.random.poisson(nbg)
        if len(fourmom) < 3*nsb:
            raise Exception("$$$ not enough bkg MC sample")
        np.savetxt(bgfile, fourmom[:3*nsb])

    ndata = 199; nbg = 133; wbg = 0.143792;
    md = "B0toD0_K3piDspi"; ch = "B0toD0_K3piDspi_Run1";
    mcfile = "toystudy/toy/BzMC3.dat"
    genfile = "toystudy/toy/BzData3.dat"
    efffile = f"../dataDDspi/{md}/EffMapCorr/EffMap_{ch}.root"
    bkgpdf = DatWeight(f"../dataDDspi/{md}/SidebandMap/BSBMap_{ch}_Smooth.root", "Hist_pos")
    #bgfile = "../dataDDspi/dat/B0toD0_K3piDspi_BSB_Run1.dat"
    bgfile = "toystudy/toy/BzBKG3.dat"
    gen_mc_from_eff_map(mcfile, ndata*100, efffile)
    if nbg*wbg != 0:
        gen_bkg_sample(bgfile, int(100*nbg*wbg), config, bkgpdf, "Bz", md)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)
    if gen_bkg_for_sfit and nbg*wbg != 0:
        fourmom = gen_bkg_sample(bgfile, int(10*nbg), config, bkgpdf, "Bz", md)
        nsb = nbg # np.random.poisson(nbg)
        if len(fourmom) < 3*nsb:
            raise Exception("$$$ not enough bkg MC sample")
        np.savetxt(bgfile, fourmom[:3*nsb])

    ndata = 835; nbg = 969; wbg = 0.106173;
    md = "B0toD0_K3piDspi"; ch = "B0toD0_K3piDspi_Run2";
    mcfile = "toystudy/toy/BzMC4.dat"
    genfile = "toystudy/toy/BzData4.dat"
    efffile = f"../dataDDspi/{md}/EffMapCorr/EffMap_{ch}.root"
    bkgpdf = DatWeight(f"../dataDDspi/{md}/SidebandMap/BSBMap_{ch}_Smooth.root", "Hist_pos")
    #bgfile = "../dataDDspi/dat/B0toD0_K3piDspi_BSB_Run2.dat"
    bgfile = "toystudy/toy/BzBKG4.dat"
    gen_mc_from_eff_map(mcfile, ndata*100, efffile)
    if nbg*wbg != 0:
        gen_bkg_sample(bgfile, int(100*nbg*wbg), config, bkgpdf, "Bz", md)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)
    if gen_bkg_for_sfit and nbg*wbg != 0:
        fourmom = gen_bkg_sample(bgfile, int(10*nbg), config, bkgpdf, "Bz", md)
        nsb = nbg # np.random.poisson(nbg)
        if len(fourmom) < 3*nsb:
            raise Exception("$$$ not enough bkg MC sample")
        np.savetxt(bgfile, fourmom[:3*nsb])

def gen_toyBp(amp, config, gen_bkg_for_sfit=True):
    ndata = 797; nbg = 407; wbg = 0.0861442;
    md = "BptoDmDspi"; ch = "BptoDmDspi_Run1";
    mcfile = "toystudy/toy/BpMC1.dat"
    genfile = "toystudy/toy/BpData1.dat"
    efffile = f"../dataDDspi/{md}/EffMapCorr/EffMap_{ch}.root"
    bkgpdf = DatWeight(f"../dataDDspi/{md}/SidebandMap/BSBMap_{ch}_Smooth.root", "Hist_pos")
    #bgfile = "../dataDDspi/dat/BptoDmDspi_BSB_Run1.dat"
    bgfile = "toystudy/toy/BpBKG1.dat"
    gen_mc_from_eff_map(mcfile, ndata*100, efffile)
    if nbg*wbg != 0:
        gen_bkg_sample(bgfile, int(100*nbg*wbg), config, bkgpdf, "Bp", md)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)
    if gen_bkg_for_sfit and nbg*wbg != 0:
        fourmom = gen_bkg_sample(bgfile, int(10*nbg), config, bkgpdf, "Bp", md)
        nsb = nbg # np.random.poisson(nbg)
        if len(fourmom) < 3*nsb:
            raise Exception("$$$ not enough bkg MC sample")
        np.savetxt(bgfile, fourmom[:3*nsb])

    ndata = 3143; nbg = 2471; wbg = 0.0554937;
    md = "BptoDmDspi"; ch = "BptoDmDspi_Run2";
    mcfile = "toystudy/toy/BpMC2.dat"
    genfile = "toystudy/toy/BpData2.dat"
    efffile = f"../dataDDspi/{md}/EffMapCorr/EffMap_{ch}.root"
    bkgpdf = DatWeight(f"../dataDDspi/{md}/SidebandMap/BSBMap_{ch}_Smooth.root", "Hist_pos")
    #bgfile = "../dataDDspi/dat/BptoDmDspi_BSB_Run2.dat"
    bgfile = "toystudy/toy/BpBKG2.dat"
    gen_mc_from_eff_map(mcfile, ndata*100, efffile)
    if nbg*wbg != 0:
        gen_bkg_sample(bgfile, int(100*nbg*wbg), config, bkgpdf, "Bp", md)
    gen_data(amp, ndata, mcfile=mcfile, genfile=genfile, Poisson_fluc=True, Nbg=nbg, wbg=wbg, bgfile=bgfile)
    gen_mc_from_eff_map(mcfile, ndata*50, efffile)
    if gen_bkg_for_sfit and nbg*wbg != 0:
        fourmom = gen_bkg_sample(bgfile, int(10*nbg), config, bkgpdf, "Bp", md)
        nsb = nbg # np.random.poisson(nbg)
        if len(fourmom) < 3*nsb:
            raise Exception("$$$ not enough bkg MC sample")
        np.savetxt(bgfile, fourmom[:3*nsb])

def gen_weigths_for_cfit(mode, config): # can be provided by dat_cfit.py
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


def set_same_D1(config): # set the total parameter of D1_2010 and D1_2007 to be the same in fitting
    try:
        vm = config.get_amplitudes()[0].vm
        vm.set_same(["B->D1_2010.DsD1_2010->D.Pi_total_0", "B->D1_2007.DsD1_2007->D.Pi_total_0"], cplx=True)
    except:
        vm = config.get_amplitude().vm

def fit_null(config):
    set_same_D1(config)
    #config.get_amplitudes()[0].vm.rp2xy_all()
    fit_result = config.fit(batch=150000, method="BFGS")
    return fit_result

def fit_Z(config, params, ZJ=0, loop=1):
    set_same_D1(config)
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
    if sfit:
        config = MultiConfig([f"toystudy/StoyBz{null}.yml", f"toystudy/StoyBp{null}.yml"], total_same=True)
    else:
        config = MultiConfig([f"toystudy/CtoyBz{null}.yml", f"toystudy/CtoyBp{null}.yml"], total_same=True)
    config.set_params(param_file)
    ampBz, ampBp = config.get_amplitudes()
    gen_toyBz(ampBz, config.configs[0])
    gen_toyBp(ampBp, config.configs[1])
    Sdnll = 0
    Cdnll = 0
    if sfit:
        print("### Null Sfit")
        Sconfig = config
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
        gen_weigths_for_cfit("Bz", config.configs[0]) # only for access to data and phsp
        gen_weigths_for_cfit("Bp", config.configs[1])
        if sfit:
            Cconfig = MultiConfig([f"toystudy/CtoyBz{null}.yml", f"toystudy/CtoyBp{null}.yml"], total_same=True)
        else: # unsolved problem here: cannot use config directly, need to load data again
            Cconfig = MultiConfig([f"toystudy/CtoyBz{null}.yml", f"toystudy/CtoyBp{null}.yml"], total_same=True) #config
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
        Sdnll, Cdnll = sig_test(sfit=False, cfit=True, null="", alternative="Z0", param_file="toystudy/params/base_s.json") # edit
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

