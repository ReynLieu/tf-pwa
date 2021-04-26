import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy import interpolate
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_index
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
                new_vals[idx] = tmp/n
        return new_vals

def get_var_from_data_dic(data_dic, mode):  
    mDPi = data_index(data_dic, ["particle", "(D, Pi)", "m"])
    mDsPi = data_index(data_dic, ["particle", "(Ds, Pi)", "m"])
    if mode == "B0toD0Dspi" or mode == "B0toD0_K3piDspi":
        mmin = 2.00440039 #1.86483+0.13957039
        mmax = 3.31131 #5.27965-1.96834
    elif mode == "BptoDmDspi":
        mmin = 2.00922039 #1.86965+0.13957039
        mmax = 3.311 #5.27934-1.96834
    dm = mmax - mmin
    mDPi_ = np.arccos(2*(mDPi-mmin)/dm - 1) / np.pi
    thetaDPi_ = 1 - data_index(data_dic, ["decay", "[B->(D, Pi)+Ds, (D, Pi)->D+Pi]", "(D, Pi)->D+Pi", "D", "ang", "beta"])/np.pi
    return [mDPi*mDPi, mDsPi*mDsPi], [mDPi_, thetaDPi_]

def main(mode):
    if mode == "Bz":
        channels = ["B0toD0Dspi_Run1", "B0toD0Dspi_Run2", "B0toD0_K3piDspi_Run1", "B0toD0_K3piDspi_Run2"]
        config = ConfigLoader("configC.yml")
    elif mode == "Bp":
        channels = ["BptoDmDspi_Run1", "BptoDmDspi_Run2"]
        config = ConfigLoader("configC_Bp.yml")

    datas = config.get_data("data")
    phsps = config.get_data("phsp")

    for data, phsp, ch in zip(datas, phsps, channels):
        md = ch[:-5]
        rn = ch[-5:]
        bkg = DatWeight(f"../dataDDspi/{md}/SidebandMap/BSBMap_{ch}_Smooth.root", "Hist_pos")
        eff = DatWeight(f"../dataDDspi/{md}/EffMapCorr/EffMap_{ch}.root", "RegDalitzEfficiency")

        Dvars, SqDvars = get_var_from_data_dic(data, md)
        save_file = f"../dataDDspi/dat_cfit/{md}_Data{rn}_value.dat"
        wbkg = bkg.weight(*SqDvars)
        weff = eff.weight(*Dvars, eff_0_correct=True)
        np.savetxt(save_file, wbkg/weff)
        print("Done saving "+save_file)

        Dvars, SqDvars = get_var_from_data_dic(phsp, md)
        save_file = f"../dataDDspi/dat_cfit/{md}_MC{rn}_value.dat"
        wbkg = bkg.weight(*SqDvars)
        weff = eff.weight(*Dvars, eff_0_correct=True)
        np.savetxt(save_file, wbkg/weff)
        print("Done saving "+save_file)


if __name__ == "__main__":
    mode = "Bz" # edit
    main(mode)
