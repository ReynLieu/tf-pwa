import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy import interpolate
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_index, data_mask
from tf_pwa.applications import gen_mc
from tf_pwa.cal_angle import prepare_data_from_decay
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
                    #print("$#@ check new_vals[idx] = 0 in dat_cfit.py")
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
def get_var_from_data_dic(data_dic, mode):  
    mDPi = data_index(data_dic, ["particle", "(D, Pi)", "m"])
    mDsPi = data_index(data_dic, ["particle", "(Ds, Pi)", "m"])
    if mode == "B0toD0Dspi" or mode == "B0toD0_K3piDspi":
        mmin = Dz + pim
        mmax = Bz - Dsp
    elif mode == "BptoDmDspi":
        mmin = Dm + pip
        mmax = Bp - Dsp
    dm = mmax - mmin
    mDPi_ = np.arccos(2*(mDPi-mmin)/dm - 1) / np.pi
    thetaDPi_ = 1 - data_index(data_dic, ["decay", "[B->(D, Pi)+Ds, (D, Pi)->D+Pi]", "(D, Pi)->D+Pi", "D", "ang", "beta"])/np.pi
    return [mDPi*mDPi, mDsPi*mDsPi], [mDPi_, thetaDPi_]

def gen_bkg_sample(bkg_file, Nsample, config, bkg_pdf, mode, md): # generate bkg sample (for cfit plot)
    if mode == "Bz":
        pf = gen_mc(Bz, [Dz, Dsp, pim], Nsample, bkg_file)
    elif mode == "Bp":
        pf = gen_mc(Bp, [Dm, Dsp, pip], Nsample, bkg_file)
    bkg_sample = prepare_data_from_decay(bkg_file, config.decay_struct)
    Dvars, SqDvars = get_var_from_data_dic(bkg_sample, md)
    maxbkgval = np.max(bkg_pdf.values)
    uni_rdm = np.random.uniform(0, maxbkgval, Nsample)
    wbkg = bkg_pdf.weight(*SqDvars)
    pf = pf.reshape(-1, 3, 4)
    bkg_four_momentum = pf[wbkg>uni_rdm]
    Nbgsample = len(bkg_four_momentum)
    bkg_four_momentum = bkg_four_momentum.reshape(-1, 4)
    np.savetxt(bkg_file, bkg_four_momentum)
    return bkg_four_momentum

def main(mode):
    if mode == "Bz":
        channels = ["B0toD0Dspi_Run1", "B0toD0Dspi_Run2", "B0toD0_K3piDspi_Run1", "B0toD0_K3piDspi_Run2"]
        bg_num = np.array([1440, 6141, 399, 2907])/3 * [0.131039, 0.103967, 0.143792, 0.106173] # edit
        config = ConfigLoader("configC.yml")
    elif mode == "Bp":
        channels = ["BptoDmDspi_Run1", "BptoDmDspi_Run2"]
        bg_num = np.array([1221, 7413])/3 * [0.0861442, 0.0554937] # edit
        config = ConfigLoader("configC_Bp.yml")

    datas = config.get_data("data")
    phsps = config.get_data("phsp")

    for data, phsp, ch, i in zip(datas, phsps, channels, range(len(channels))):
        md = ch[:-5]
        rn = ch[-5:]
        bkg = DatWeight(f"../dataDDspi/{md}/SidebandMap/BSBMap_{ch}_Smooth.root", "Hist_pos")
        eff = DatWeight(f"../dataDDspi/{md}/EffMapCorr/EffMap_{ch}.root", "RegDalitzEfficiency")

        # calculate bkg_value, eff_value for data
        Dvars, SqDvars = get_var_from_data_dic(data, md)
        save_file = f"../dataDDspi/dat_cfit/{md}_Data{rn}_value.dat"
        wbkg = bkg.weight(*SqDvars)
        weff = eff.weight(*Dvars, eff_0_correct=True)
        np.savetxt(save_file, wbkg/weff)
        print("Done saving "+save_file)

        # calculate bkg_value, eff_value for phsp MC
        Dvars, SqDvars = get_var_from_data_dic(phsp, md)
        save_file = f"../dataDDspi/dat_cfit/{md}_MC{rn}_value.dat"
        wbkg = bkg.weight(*SqDvars)
        weff = eff.weight(*Dvars, eff_0_correct=True)
        np.savetxt(save_file, wbkg/weff)
        print("Done saving "+save_file)

        bkg_file = f"../dataDDspi/dat_cfit/{md}{rn}_BKGsample.dat"
        fourmom = gen_bkg_sample(
            bkg_file=bkg_file, Nsample=int(100*bg_num[i]), config=config, bkg_pdf=bkg, mode=mode, md=md
            )
        Nbgsample = len(fourmom)//3 # one event has three lines
        print(f"Done saving {bkg_file} with {Nbgsample} events, the weight should be {bg_num[i]/Nbgsample}")


if __name__ == "__main__":
    main("Bz") # edit
    main("Bp") # edit
