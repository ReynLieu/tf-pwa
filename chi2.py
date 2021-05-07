#!/usr/bin/env python3
import os.path
import sys

import matplotlib.colors as mcolors
import matplotlib.patches as mpathes
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np

from tf_pwa.adaptive_bins import AdaptiveBound, cal_chi2
from tf_pwa.angle import kine_max, kine_min
from tf_pwa.config_loader import ConfigLoader
from tf_pwa.data import data_index, data_to_numpy
from tf_pwa.experimental import extra_amp, extra_data
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
    adapter, data, phsp, bg=None, n_fp=0
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
    return cal_chi2(numbers, n_fp), numbers


def draw_dalitz(data_cut, bound, numbers):
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
        data_cut[0], data_cut[1], c="black", s=1.0, marker='.', alpha=0.5
    )
    ah.set_zorder(100)
    # ax.scatter(data_cut[0]**2, data_cut[1]**2, s=1, c="red")
    ## using your own mass
    m0, mb, mc, md = 5.27963, 2.01026, 1.86483, 0.493677
    # print(ah)
    sbc_min, sbc_max = (mb + mc) ** 2, (m0 - md) ** 2
    sbd_min, sbd_max = (mb + md) ** 2, (m0 - mc) ** 2
    sbc = np.linspace(sbc_min, sbc_max, 1000)
    ax.plot(sbc, kine_max(sbc, m0, mc, mb, md), color="grey")
    ax.plot(sbc, kine_min(sbc, m0, mc, mb, md), color="grey")

    ax.set_xlim((np.min(data_cut[0]), np.max(data_cut[0])))
    ax.set_ylim((np.min(data_cut[1]), np.max(data_cut[1])))
    ax.set_xlabel("$M_{D^{0}K^{+}}^2$ GeV$^2$")
    ax.set_ylabel("$M_{D^{*-}D^{0}}^2$ GeV$^2$")
    x_major_locator=MultipleLocator(0.5)
    x_minor_locator=MultipleLocator(0.1)
    y_major_locator=MultipleLocator(1)
    y_minor_locator=MultipleLocator(0.2)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    plt.tick_params(top='on', right='on', which='both')

    gradient = np.linspace(-max_chi, max_chi, 256)
    gradient = np.vstack((gradient, gradient))
    img = ax.imshow(gradient, aspect='auto', cmap=my_cmap)
    fig.colorbar(img) # ah[-1])
    save_path = "figs/BW_2Dpull.png" # edit here
    plt.savefig(save_path, dpi=200)
    print("Done saving", save_path)


def main():
    data, phsp, bg = load_root_data("save/BWbase/BWbase.root") # edit here
    n_fp = 27 # edit here

    data_cut = np.array([data["m_R_CD"]**2, data["m_R_BC"]**2])
    adapter = AdaptiveBound(data_cut, [[3, 2]]*3)
    bound = adapter.get_bounds()

    phsp_cut = np.array([phsp["m_R_CD_MC"]**2, phsp["m_R_BC_MC"]**2, phsp["MC_total_fit"]])
    # print(np.sum(bg["sideband_weights"])
    bg_cut = np.array([bg["m_R_CD_sideband"]**2, bg["m_R_BC_sideband"]**2, bg["sideband_weights"]])

    _, numbers = cal_chi2_config(adapter, data_cut, phsp_cut, bg_cut, n_fp)
    # print(numbers)
    draw_dalitz(data_cut, bound, numbers)


if __name__ == "__main__":
    if not os.path.exists("figs"):
        os.mkdir("figs")
    main()
