#!/usr/bin/env python3

# avoid using Xwindow
import matplotlib

matplotlib.use("agg")

from tf_pwa.config_loader import ConfigLoader, MultiConfig
from tf_pwa.variable import VarsManager
from pprint import pprint
from tf_pwa.utils import error_print
import tensorflow as tf
import json
import time
import numpy as np
from fit import *


# examples of custom particle model
from tf_pwa.amp import simple_resonance
from tf_pwa.experimental import extra_amp, extra_data


def json_print(dic):
    """print parameters as json"""
    s = json.dumps(dic, indent=2)
    print(s, flush=True)


def fit(config, config0, vm, method="BFGS", loop=100):

    fit_results = []
    nlls = []
    statuses= []
    config0.get_params() # in case new vm is generated when get_amplitude()
    config.set_params("init.json")
    for i in range(loop):
        print("\nusing RANDOM parameters", flush=True)
        # try to fit
        config.fit(batch=65000, method=method)
        fit_result = config0.fit(batch=65000, method=method)
        json_print(fit_result.params)

        fit_results.append(fit_result)
        nlls.append(fit_result.min_nll)
        statuses.append(fit_result.success)
        print("$$$NLL\n", nlls)
        print("$$$status\n",i, statuses)
        # reset parameters
        try:
            config.reinit_params()
            vm.set("NR(DK)PS_a", np.random.uniform(0,1))
            vm.set("A->Ds2_2573p.BDs2_2573p->C.DB->E.F_total_0r", 1)
            vm.set("A->Ds2_2573p.BDs2_2573p->C.DB->E.F_total_0i", 0)
            vm.set("Ds1_2700_2860_mass1", 2.7083)
            vm.set("Ds1_2700_2860_mass2", 2.91)
            vm.set("Ds1_2700_2860_G1a", 0.063)
            vm.set("Ds1_2700_2860_G1r", 0.91)
            vm.set("Ds1_2700_2860_G2a", 0.076)
            vm.set("Ds1_2700_2860_G2r", 1.10)
            vm.set("DK(1-)SP_point_4r", 1)
            vm.set("DK(1-)SP_point_4i", 0)
            vm.set("DK(1-)PP_point_3r", 1)
            vm.set("DK(1-)PP_point_3i", 0)
            vm.set("DK(1-)DP_point_2r", 1)
            vm.set("DK(1-)DP_point_2i", 0)

        except Exception as e:
            print(e)




def main():
    """entry point of fit. add some arguments in commond line"""
    import argparse

    parser = argparse.ArgumentParser(description="simple fit scripts")
    parser.add_argument("--no-GPU", action="store_false", default=True, dest="has_gpu")
    #parser.add_argument("-c", "--config", default="config.yml", dest="config")
    #parser.add_argument("-i", "--init_params", default="init_params.json", dest="init")
    parser.add_argument("-m", "--method", default="BFGS", dest="method")
    parser.add_argument("-l", "--loop", type=int, default=100, dest="loop")
    results = parser.parse_args()
    if results.has_gpu:
        devices = "/device:GPU:0"
    else:
        devices = "/device:CPU:0"
    with tf.device(devices):
        vm = VarsManager()
        config = ConfigLoader("config.yml", vm=vm)
        config0 = ConfigLoader("config0.yml", vm=vm)
        fit_result = fit(config, config0, vm, results.method, results.loop)


if __name__ == "__main__":
    main()
