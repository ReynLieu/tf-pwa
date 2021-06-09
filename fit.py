#!/usr/bin/env python3

import csv
import json
import time
from pprint import pprint

# avoid using Xwindow
import matplotlib

matplotlib.use("agg")

import tensorflow as tf

# examples of custom particle model
from tf_pwa.amp import simple_resonance, register_particle, Particle
from tf_pwa.config_loader import ConfigLoader, MultiConfig
from tf_pwa.experimental import extra_amp, extra_data
from tf_pwa.utils import error_print, tuple_table


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
        mass = self.get_mass()
        width = self.get_width()
        q = data_c["|q|"]
        q0 = 0.042580705388 # get_p(2.00685, 1.86483, 0.1349768)
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


def fit(config, vm, init_params="", method="BFGS", loop=1, maxiter=500):
    """
    simple fit script
    """
    # load config.yml
    # config = ConfigLoader(config_file)

    # load data
    all_data = config.get_all_data()
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
            #vm.rp2xy_all()
            vm.std_polar_all()
            fit_result = config.fit(
                batch=65000, method=method, maxiter=maxiter
            )
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
        fit_result.save_as("final_params.json")
        pprint(fit_error)

        print("\n########## fit results:")
        print("Fit status: ", fit_result.success)
        print("Minimal -lnL = ", fit_result.min_nll)
        for k, v in config.get_params().items():
            print(k, error_print(v, fit_error.get(k, None)))

    return fit_result

from frac_table import frac_table
from tf_pwa.data import data_cut
res_curvestyle = {"D0_2300":"lime", "D0_2900":"y"}
def write_some_results(config, fit_result, save_root=False):
    # plot partial wave distribution
    res = None
    config.plot_partial_wave(fit_result, plot_pull=True, save_root=save_root, smooth=False, save_pdf=True, res=res, res_curvestyle=res_curvestyle)

    # calculate fit fractions
    phsp_noeff = config.get_phsp_noeff()
    fit_frac, err_frac = config.cal_fitfractions({}, phsp_noeff)

    print("########## fit fractions")
    fit_frac_string = ""
    for i in fit_frac:
        if isinstance(i, tuple):
            name = "{}x{}".format(*i)
        else:
            name = i
        fit_frac_string += "{} {}\n".format(
            name, error_print(fit_frac[i], err_frac.get(i, None))
        )
    print(fit_frac_string)
    save_frac_csv("fit_frac.csv", fit_frac)
    save_frac_csv("fit_frac_err.csv", err_frac)
    from frac_table import frac_table
    frac_table(fit_frac_string)
    # chi2, ndf = config.cal_chi2(mass=["R_BC", "R_CD"], bins=[[2,2]]*4)

def write_some_results_combine(config, fit_result, save_root=False):

    from tf_pwa.applications import fit_fractions

    for i, c in enumerate(config.configs):
        res = None
        '''if i is 0:
            res = ["D1_2010","D2_2460","D1_2600",["NR_DPi0","D0_2400m"],"X0"]
        elif i is 1:
            res = ["D1_2007","D2_2460","D1_2600",["NR_DPi0","D0_2400o"],"X0"]'''
        c.plot_partial_wave(
            fit_result, prefix="figure/", plot_pull=True, save_root=save_root, smooth=False, save_pdf=True, res=res, res_curvestyle=res_curvestyle
        )
        '''cut_string = "(m>2.7)"
        data, phsp, bg, _ = c.get_all_data()
        data = [data_cut(i, cut_string, {"m": ("particle", "(D, Pi)", "m")}) for i in data]
        phsp = [data_cut(i, cut_string, {"m": ("particle", "(D, Pi)", "m")}) for i in phsp]
        bg = [data_cut(i, cut_string, {"m": ("particle", "(D, Pi)", "m")}) for i in bg]
        #bg = [None, None]
        c.plot_partial_wave(fit_result, prefix="figure/", plot_pull=True, smooth=False, data=data, phsp=phsp, bg=bg, save_pdf=True)
        '''

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
        fit_frac_string = ""
        for i in fit_frac:
            if isinstance(i, tuple):
                name = "{}x{}".format(*i)  # interference term
            else:
                name = i  # fit fraction
            fit_frac_string += "{} {}\n".format(
                name, error_print(fit_frac[i], err_frac.get(i, None))
            )
        print(fit_frac_string)
        save_frac_csv(f"fit_frac{it}.csv", fit_frac)
        save_frac_csv(f"fit_frac{it}_err.csv", err_frac)
        from frac_table import frac_table
        frac_table(fit_frac_string)


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
        try:
            vm.set_same(["B->D0_2400m.DsD0_2400m->D.Pi_total_0", "B->D0_2400o.DsD0_2400o->D.Pi_total_0"], cplx=True)
        except:
            pass
        fit_result = fit(
            config, vm, results.init, results.method, results.loop, results.maxiter
        )
        if isinstance(config, ConfigLoader):
            write_some_results(
                config, fit_result, save_root=results.save_root
            )
        else:
            write_some_results_combine(
                config, fit_result, save_root=results.save_root
            )


if __name__ == "__main__":
    write_run_point()
    main()
