#!/usr/bin/env python3
import numpy as np
from fit import *


def main(mode, ntries=10, cs="cfit"): # random search for new resonance
    with tf.device("/device:GPU:0"):
        if cs == "cfit":
            config = load_config("configC.yml,configC_Bp.yml", True)
        elif cs == "sfit":
            config = load_config("config.yml,config_Bp.yml", True)
        try:
            vm = config.get_amplitudes()[0].vm
            vm.set_same(["B->D1_2010.DsD1_2010->D.Pi_total_0", "B->D1_2007.DsD1_2007->D.Pi_total_0"], cplx=True)
        except:
            vm = config.get_amplitude().vm
        #vm.rp2xy_all()
        
        if mode == "Dstar":
            for mass in np.linspace(2.1, 3.2, ntries):
                vm.set("Dstar_mass", mass)
                vm.set("Dstar_width", 0.1)
                vm.set("B->Dstar.DsDstar->D.Pi_total_0r", 0)
                fit_result = fit(
                    config, vm, f"save/base_{cs[0]}/final_params.json", "BFGS", 1, 2000
                )
        elif mode == "DsPi":
            for mass in np.linspace(2.2, 3.3, ntries):
                vm.set("DsPi_mass", mass)
                vm.set("DsPi_width", 0.1)
                vm.set("B->DsPi.DDsPi->Ds.Pi_total_0r", 0)
                fit_result = fit(
                    config, vm, f"save/baseZ0_{cs[0]}/final_params.json", "BFGS", 1, 2000
                )
        elif mode == "DDs":
            for mass in np.linspace(4.0, 5.1, ntries):
                vm.set("DDs_mass", mass)
                vm.set("DDs_width", 0.1)
                vm.set("B->DDs.PiDDs->D.Ds_total_0r", 0)
                fit_result = fit(
                    config, vm, f"save/baseZ0_{cs[0]}/final_params.json", "BFGS", 1, 2000
                )


if __name__ == "__main__":
    mode = "Dstar" # "Dstar", "DsPi", "DDs"
    main(mode, 10, "cfit")
