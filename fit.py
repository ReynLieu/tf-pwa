#!/usr/bin/env python3

import csv
import json
import time
from pprint import pprint

# avoid using Xwindow
import matplotlib

matplotlib.use("agg")

import tensorflow as tf
import numpy as np

# examples of custom particle model
from tf_pwa.amp import simple_resonance, register_particle, Particle
from tf_pwa.config_loader import ConfigLoader, MultiConfig
from tf_pwa.experimental import extra_amp, extra_data
from tf_pwa.utils import error_print, tuple_table
from tf_pwa.data import data_cut


@register_particle("exp2")
class ParticleExp(Particle):
    def init_params(self):
        self.a = self.add_var("a")

    def get_amp(self, data, _data_c=None, **kwargs):
        mass = data["m"]
        zeros = tf.zeros_like(mass)
        a = tf.abs(self.a())
        return tf.complex(tf.exp(-a * (mass * mass - 4.0)), zeros) # - 4.0272863761

# D0 1.86483 pi0 0.1349768
from tf_pwa.breit_wigner import BWR, Bprime
@register_particle("BW2007")
class ParticleBW2007(Particle):
    def get_amp(self, data, data_c=None, **kwargs):
        mass = self.get_mass() # 2.00685
        width = self.get_width()
        q = data_c["|q|"]
        q0 = 0.042580705388 # get_p(mass, 1.86483, 0.1349768)
        '''mmin = 1.86965 + 0.13957039
        mmax = 5.27934 - 1.96834
        meff = 2.358391539393489 # mmin + (mmax-mmin)/2 * (1+np.tanh((mass-(mmin+mmax)/2)/(mmax-mmin)))'''
        #q0 = 0.42485215283569866 # get_p(meff, 1.86483, 0.1349768)

        BWamp = BWR(data["m"], mass, width, q, q0, L=1, d=3.0)
        barrier_factor = q * Bprime(1, q, q0, d=3.0)
        return BWamp * tf.cast(barrier_factor, BWamp.dtype)


def json_print(dic):
    """print parameters as json"""
    s = json.dumps(dic, indent=2)
    print(s, flush=True)


def load_config(config_file="config.yml", total_same=False):
    config_files = config_file.split(",")
    if len(config_files) == 1:
        return ConfigLoader(config_files[0])
    return MultiConfig(config_files, total_same=total_same)


cut_string = ""#"(m<2.7)|(m>3.1)"
cut_varidx = {"m": ("particle", "(D, Pi)", "m")}
def fit(config, vm, init_params="", method="BFGS", loop=1, maxiter=500, xycoord=0):
    """
    simple fit script
    """
    # load data
    if not cut_string:
        all_data = config.get_all_data()
    else:
        datas1, datas2 = config.get_all_data()
        cut_datas1 = []; cut_datas2 = []
        for data in datas1:
            if data[0] is not None:
                data = [data_cut(i, cut_string, cut_varidx) for i in data]
            cut_datas1.append(data)
        for data in datas2:
            if data[0] is not None:
                data = [data_cut(i, cut_string, cut_varidx) for i in data]
            cut_datas2.append(data)
        all_data = [cut_datas1, cut_datas2]
        
    if xycoord == 1: # use xy init_params and fit in xy
        vm.rp2xy_all()
    fit_results = []
    '''if maxiter is 0:
        try:
            amp = config.get_amplitude()
            amp.cached_fun = amp.decay_group.sum_amp # turn off use_tf_function
        except:
            amp_list = config.get_amplitudes()
            for amp in amp_list:
                amp.cached_fun = amp.decay_group.sum_amp # turn off use_tf_function'''
    for i in range(loop):
        # set initial parameters if have
        if config.set_params(init_params):
            print("using {}".format(init_params))
        else:
            print("\nusing RANDOM parameters", flush=True)
        # try to fit
        try:
            if xycoord == 2: # fit in xy
                vm.rp2xy_all()
            fit_result = config.fit(
                datas=all_data, batch=65000, method=method, maxiter=maxiter
            )
            vm.std_polar_all()
        except KeyboardInterrupt:
            config.save_params("break_params.json")
            raise
        except Exception as e:
            print(e)
            config.save_params("break_params.json")
            raise
        fit_results.append(fit_result)
        # reset parameters
        try:
            config.reinit_params() # a bug found when loop>1: for variable with bound, the initial NLL does not have bound transform, but the following fit has no problem
        except Exception as e:
            print(e)

    fit_result = fit_results.pop()
    for i in fit_results:
        if i.success:
            if not fit_result.success or fit_result.min_nll > i.min_nll:
                fit_result = i

    config.set_params(fit_result.params)
    print("\n ##### The best result:")
    json_print(fit_result.params)
    fit_result.save_as("final_params.json")

    # calculate parameters error
    if maxiter is not 0:
        fit_error = config.get_params_error(fit_result, batch=13000)
        fit_result.set_error(fit_error)
        #fit_result.save_as("final_params.json")
        pprint(fit_error)

        print("\n########## fit results:")
        print("Fit status: ", fit_result.success)
        print("Minimal -lnL = ", fit_result.min_nll)
        for k, v in config.get_params().items():
            print(k, error_print(v, fit_error.get(k, None)))

    return fit_result

from frac_table import frac_table
res_curvestyle = {"D0_2300":"lime", "D0_2900":"y"}
def write_some_results(config, fit_result, save_root=False):
    # plot partial wave distribution
    res = None
    config.plot_partial_wave(fit_result, plot_pull=True, save_root=save_root, smooth=False, save_pdf=True, res=res, res_curvestyle=res_curvestyle)

    # calculate fit fractions
    phsp_noeff = config.get_phsp_noeff()
    fit_frac, err_frac = config.cal_fitfractions({}, phsp_noeff)
    fitfrac = {}; errfrac = {}; 

    print("########## fit fractions")
    fit_frac_string = ""
    for i in fit_frac:
        if isinstance(i, tuple):
            name = "{}x{}".format(*i)
        else:
            name = i
            fitfrac[name] = fit_frac[name]
            errfrac[name] = err_frac[name]
        fit_frac_string += "{} {}\n".format(
            name, error_print(fit_frac[i], err_frac.get(i, None))
        )
    print(fit_frac_string)
    save_frac_csv("fit_frac.csv", fit_frac)
    save_frac_csv("fit_frac_err.csv", err_frac)
    from frac_table import frac_table
    frac_table(fit_frac_string)
    # chi2, ndf = config.cal_chi2(mass=["R_BC", "R_CD"], bins=[[2,2]]*4)
    fit_result.set_fitfrac(fitfrac, errfrac)
    return fit_result

def write_some_results_combine(config, fit_result, save_root=False):

    from tf_pwa.applications import fit_fractions

    for i, c in enumerate(config.configs):
        res = None
        '''if i is 0:
            res = ["D1_2010","D2_2460","D1_2600",["NR_DPi0","D0_2400m"],"X0"]
        elif i is 1:
            res = ["D1_2007","D2_2460","D1_2600",["NR_DPi0","D0_2400o"],"X0"]'''
        if not cut_string:
            c.plot_partial_wave(
                fit_result, prefix="figure/", plot_pull=True, save_root=save_root, smooth=False, save_pdf=True, res=res, res_curvestyle=res_curvestyle
            )
        else:
            data, phsp, bg, _ = c.get_all_data()
            data = [data_cut(i, cut_string, cut_varidx) for i in data]
            phsp = [data_cut(i, cut_string, cut_varidx) for i in phsp]
            bg = [data_cut(i, cut_string, cut_varidx) for i in bg] # [None, None]
            c.plot_partial_wave(fit_result, prefix="figure/", plot_pull=True, smooth=False, data=data, phsp=phsp, bg=bg, save_pdf=True)
        
    fitfrac = []; errfrac = []; 
    for it, config_i in enumerate(config.configs):
        print("########## fit fractions {}:".format(it))
        print(f"nll{it}", config_i.get_fcn()({}).numpy())
        mcdata = config_i.get_phsp_noeff()
        fit_frac, err_frac = fit_fractions(
            config_i.get_amplitude(),
            mcdata,
            config.inv_he,
            fit_result.params,
        )
        fitfrac.append({})
        errfrac.append({})
        fit_frac_string = ""
        for i in fit_frac:
            if isinstance(i, tuple):
                name = "{}x{}".format(*i)  # interference term
            else:
                name = i  # fit fraction
                fitfrac[it][name] = fit_frac[name]
                errfrac[it][name] = err_frac[name]
            fit_frac_string += "{} {}\n".format(
                name, error_print(fit_frac[i], err_frac.get(i, None))
            )
        print(fit_frac_string)
        save_frac_csv(f"fit_frac{it}.csv", fit_frac)
        save_frac_csv(f"fit_frac{it}_err.csv", err_frac)
        from frac_table import frac_table
        frac_table(fit_frac_string)
    fit_result.set_fitfrac(fitfrac, errfrac)
    return fit_result


def save_frac_csv(file_name, fit_frac):
    table = tuple_table(fit_frac)
    with open(file_name, "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(table)


def write_run_point():
    """ write time as a point of fit start"""
    with open(".run_start", "w") as f:
        localtime = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())
        )
        f.write(localtime)


def main():
    """entry point of fit. add some arguments in commond line"""
    import argparse

    parser = argparse.ArgumentParser(description="simple fit scripts")
    parser.add_argument(
        "--no-GPU", action="store_false", default=True, dest="has_gpu"
    )
    parser.add_argument(
        "-c", "--config", default="config.yml", dest="config"
    )
    parser.add_argument(
        "-i", "--init_params", default="init_params.json", dest="init"
    )
    parser.add_argument(
        "-m", "--method", default="BFGS", dest="method"
    )
    parser.add_argument(
        "-l", "--loop", type=int, default=1, dest="loop"
    )
    parser.add_argument(
        "-x", "--maxiter", type=int, default=2000, dest="maxiter"
    )
    parser.add_argument(
        "-r", "--save_root", default=False, dest="save_root"
    )
    parser.add_argument(
        "--total-same", action="store_true", default=True, dest="total_same"
    )
    parser.add_argument(
        "-y", "--xycoord", type=int, default=0, dest="xycoord"
    )
    results = parser.parse_args()
    if results.has_gpu:
        devices = "/device:GPU:0"
    else:
        devices = "/device:CPU:0"
    with tf.device(devices):
        config = load_config(results.config, results.total_same)
        try:
            vm = config.get_amplitudes()[0].vm
            vm.set_same(["B->D1_2010.DsD1_2010->D.Pi_total_0", "B->D1_2007.DsD1_2007->D.Pi_total_0"], cplx=True)
        except:
            vm = config.get_amplitude().vm
        #vm.rp2xy_all()

        fit_result = fit(
            config, vm, results.init, results.method, results.loop, results.maxiter, results.xycoord
        )
        if isinstance(config, ConfigLoader):
            fit_result = write_some_results(
                config, fit_result, save_root=results.save_root
            )
        else:
            fit_result = write_some_results_combine(
                config, fit_result, save_root=results.save_root
            )
        fit_result.save_as("final_params.json")


if __name__ == "__main__":
    write_run_point()
    main()
