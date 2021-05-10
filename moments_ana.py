import uproot
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legval
from tf_pwa.histogram import Hist1D

def legendre_poly(x,L):
    x=np.array(x)
    xpoly = []
    for i in range(L,-1,-2):
        xpoly.append(x**i)
    xpoly = np.array(xpoly)
    prefactor = np.sqrt((2*L+1)/2)
    if L == 0:
        return prefactor * np.ones_like(x)
    elif L == 1:
        return prefactor * x
    elif L == 2:
        return prefactor/2 * np.einsum("ij,i->j",xpoly,[3,-1])
    elif L == 3:
        return prefactor/2 * np.einsum("ij,i->j",xpoly,[5,-3])
    elif L == 4:
        return prefactor/8 * np.einsum("ij,i->j",xpoly,[35,-30,3])
    elif L == 5:
        return prefactor/8 * np.einsum("ij,i->j",xpoly,[63,-70,15])
    elif L == 6:
        return prefactor/16 * np.einsum("ij,i->j",xpoly,[231,-315,105,-5])
    elif L == 7:
        return prefactor/16 * np.einsum("ij,i->j",xpoly,[429,-693,315,-35])
    elif L == 8:
        return prefactor/128 * np.einsum("ij,i->j",xpoly,[6435,-12012,6930,-1260,35])


def get_data(tree, tail="", weight=None, mode="Bp"): # edit
    costheta = tree.get(f"{mode}R_DPi_D_cos_beta_{tail}").array()
    mDstD = tree.get(f"m_{mode}R_DPi{tail}").array()
    theta_ij = np.arccos(tree.get(f"{mode}R_DPi_D_cos_beta_{tail}").array())
    if weight is None:
        weights = np.ones_like(mDstD)
    else:
        weights = tree.get(weight).array()
    return np.stack([mDstD, theta_ij, costheta, weights])

def plot1(data):
    plt.clf()
    plt.scatter(data[2], data[0], s=1, c="black")
    plt.ylabel("M_(DPi)/GeV")
    plt.xlabel(r"$\cos \theta$")
    plt.title(r"$\cos \theta vs M(Dpi)$")
    plt.savefig("2d_m_theta.png")

def data_sub(data, bg):
    return np.stack([
        np.concatenate([data[0], bg[0]]),
        np.concatenate([data[1], bg[1]]),
        np.concatenate([data[2], bg[2]]),
        np.concatenate([data[3], -bg[3]]),
    ])


def cal_moment_weight(theta, weight, order=0):
    x = np.cos(theta)
    coef = np.zeros((order+1,))
    coef[order] = 1
    #pl = legval(-x, coef)
    pl = legendre_poly(-x, len(coef)-1)
    ret = weight*pl#/np.sum(weight)
    return ret

def plot_moment(data, fitted, order=0, prefix=""):
    w_data = cal_moment_weight(data[1], data[3], order)
    w_fitted = cal_moment_weight(fitted[1], fitted[3], order)
    plt.clf()
    x_min, x_max = np.min(fitted[0]**2)-1e-6, np.max(fitted[0]**2)+1e-6
    hist1 = Hist1D.histogram(data[0]**2, weights=w_data, range=(x_min, x_max), bins=50)
    hist2 = Hist1D.histogram(fitted[0]**2, weights=w_fitted, range=(x_min, x_max), bins=50)
    ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1)
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3, sharex=ax2)
    hist1.draw(ax, label="data")
    hist1.draw_error(ax)
    hist2.draw(ax, label="fit")
    hist2.draw_error(ax)
    (hist1-hist2).draw_pull(ax2)
    ax2.set_xlabel(r"$M_{D^-\pi^+}$/GeV") # edit
    ax.set_ylabel(f"$\\langle Y_{order} \\rangle$")
    ax.minorticks_on()
    ax.tick_params(axis='y', which='minor', left=False)
    ax2.set_ylabel("pull")
    ax2.set_ylim((-5,5))
    ax2.minorticks_on()


def main():
    mode = "Bp" # edit
    with uproot.open("figure/variables_com.root") as f: # edit
        data = get_data(f.get("data"), "", "data_weights", mode)
        bg = get_data(f.get("sideband"), "_sideband", "sideband_weights", mode)
        fitted = get_data(f.get("fitted"), "_MC", "MC_total_fit", mode)
        
    sub_data = data_sub(data, bg)
    
    #cut1 = np.abs(sub_data[2]) > 0.5
    #cut2 = np.abs(fitted[2]) > 0.5

    plot1(data)
    for i in range(9):
        plot_moment(sub_data, fitted, i)
        plt.title(f"moment {i}")
        plt.savefig(f"figs/{mode}_Dpi_moment_{i}.png")
    '''for i in range(8):
        plot_moment(sub_data[:,cut1], fitted[:,cut2], i)
        plt.title("moment {}: $|\\cos\\theta| > 0.5$".format(i))
        plt.savefig(f"g0_Dpi_moment_{i}.png")
    for i in range(8):
        plot_moment(sub_data[:,~cut1], fitted[:,~cut2], i)
        plt.title("moment {}: $|\\cos\\theta| <= 0.5$".format(i))
        plt.savefig(f"l0_Dpi_moment_{i}.png")'''
    
    


if __name__ == "__main__":
    main()
