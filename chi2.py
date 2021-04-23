#!/usr/bin/env python3
import os.path
import sys

import matplotlib.colors as mcolors
import matplotlib.patches as mpathes
import matplotlib.pyplot as plt
import numpy as np

from tf_pwa.adaptive_bins import AdaptiveBound, cal_chi2
from tf_pwa.angle import kine_max, kine_min
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_index, data_to_numpy
from tf_pwa.experimental import extra_amp, extra_data
from fit import ParticleExp
import uproot

def load_root_data(file_name):
    with uproot.open(file_name) as f:
        data = f.get("data")
        data = {k.decode(): data.get(k).array() for k in data.keys()}
        phsp = f.get("fitted")
        phsp = {k.decode(): phsp.get(k).array() for k in phsp.keys()}
        bg = f.get("sideband")
        bg = {k.decode(): bg.get(k).array() for k in bg.keys()}
    return data, phsp, bg


def cal_chi2_config(
    adapter, data, phsp, bg=None
):
    phsps = adapter.split_data(phsp)
    datas = adapter.split_data(data)
    bound = adapter.get_bounds()
    if bg is not None:
        bgs = adapter.split_data(bg)
    int_norm = 1
    print("int norm:", int_norm)
    numbers = []
    for i, bnd in enumerate(bound):
        min_x, min_y = bnd[0]
        max_x, max_y = bnd[1]
        ndata = datas[i].shape[-1]
        nmc = np.sum(phsps[i][2]) * int_norm
        if bg is not None:
            nmc += np.sum(bgs[i][2])
        numbers.append((ndata, nmc))
    return cal_chi2(numbers, 0), numbers


def draw_dalitz(data_cut, bound, numbers, mode):
    fig, ax = plt.subplots()
    #my_cmap = plt.get_cmap("jet")
    clist = [(0,"darkblue"), (0.5,"white"), (1, "darkred")]
    my_cmap = mcolors.LinearSegmentedColormap.from_list("name", clist)
    # print(my_cmap(1.0))
    # my_cmap.set_under('w', 1)
    
    max_chi = 5.
    for i, bn in enumerate(zip(bound, numbers)):
        bnd, ns = bn
        ndata, nmc = ns
        chi = (ndata-nmc)/np.sqrt(np.abs(ndata))
        max_chi = max(abs(chi), max_chi)

    for i, bn in enumerate(zip(bound, numbers)):
        bnd, ns = bn
        ndata, nmc = ns
        min_x, min_y = bnd[0]
        max_x, max_y = bnd[1]
        chi = (ndata-nmc)/np.sqrt(np.abs(ndata))
        # print(chi, ndata, nmc)
        c = int(256*(chi / 2/max_chi + 0.5))
        if True: 
            color = my_cmap(c)
        else:
            color = "none"
        # print(c, color)
        rect = mpathes.Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            linewidth=0.1,
            facecolor=color,
            edgecolor="white",
            alpha=1.
        )  # cmap(weights[i]/max_weight))
        ax.add_patch(rect)

    ah = ax.scatter(
        data_cut[0], data_cut[1], c="black", s=1.0
    )
    ah.set_zorder(100)
    # ax.scatter(data_cut[0]**2, data_cut[1]**2, s=1, c="red")
    ## using your own mass
    if mode == "Bz":
        m0, mb, mc, md = 5.27963, 0.13957039, 1.86483, 1.96834
    elif mode == "Bp":
        m0, mb, mc, md = 5.27934, 0.13957039, 1.86965, 1.96834
    # print(ah)
    sbc_min, sbc_max = (mb + mc) ** 2, (m0 - md) ** 2
    sbd_min, sbd_max = (mb + md) ** 2, (m0 - mc) ** 2
    sbc = np.linspace(sbc_min, sbc_max, 1000)
    ax.plot(sbc, kine_max(sbc, m0, mc, mb, md), color="grey")
    ax.plot(sbc, kine_min(sbc, m0, mc, mb, md), color="grey")

    ax.set_xlim((np.min(data_cut[0]), np.max(data_cut[0])))
    ax.set_ylim((np.min(data_cut[1]), np.max(data_cut[1])))
    if mode == "Bz":
        ax.set_xlabel("$M_{\\bar D^{0}\\pi^{-}}^2$ GeV$^2$")
        ax.set_ylabel("$M_{D_{s}^{+}\\pi^{-}}^2$ GeV$^2$")
    elif mode == "Bp":
        ax.set_xlabel("$M_{D^{-}\\pi^{+}}^2$ GeV$^2$")
        ax.set_ylabel("$M_{D_{s}^{+}\\pi^{+}}^2$ GeV$^2$")
    gradient = np.linspace(-5, 5, 256)
    gradient = np.vstack((gradient, gradient))
    img = ax.imshow(gradient, aspect='auto', cmap=my_cmap)
    fig.colorbar(img) # ah[-1])
    plt.savefig(f"figs/{mode}2Dpull.png", dpi=200)


def main():
    mode = "Bz" # edit here
    data, phsp, bg = load_root_data("trash/figure/variables_com.root") # edit here

    data_cut = np.array([data[f"m_{mode}R_DPi"]**2, data[f"m_{mode}R_DsPi"]**2])
    adapter = AdaptiveBound(data_cut, [[2, 2]]*3)
    bound = adapter.get_bounds()

    phsp_cut = np.array([phsp[f"m_{mode}R_DPi_MC"]**2, phsp[f"m_{mode}R_DsPi_MC"]**2, phsp["MC_total_fit"]])
    # print(np.sum(bg["sideband_weights"])
    bg_cut = np.array([bg[f"m_{mode}R_DPi_sideband"]**2, bg[f"m_{mode}R_DsPi_sideband"]**2, bg["sideband_weights"]])

    _, numbers = cal_chi2_config(adapter, data_cut, phsp_cut, bg_cut)
    # print(numbers)
    draw_dalitz(data_cut, bound, numbers, mode)


if __name__ == "__main__":
    main()
