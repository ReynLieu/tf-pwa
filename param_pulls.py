from toy_fit import *
from tf_pwa.variable import VarsManager


def param_pulls(i, base="", param_file=None, sfit=True, cfit=True, **kwargs):
    if param_file is None:
        param_file = f"toystudy/params/base{base}_s.json"
    vm = VarsManager()
    vm.rp2xy_all() # fit in xy
    Sconfig = MultiConfig([f"toystudy/StoyBz{base}.yml", f"toystudy/StoyBp{base}.yml"], total_same=True, vm=vm)
    Sconfig.set_params(param_file)
    ampBz, ampBp = Sconfig.get_amplitudes()
    gen_toyBz(ampBz)
    gen_toyBp(ampBp)
    if i == 0:
        for v in vm.trainable_vars:
            fp_s[v]=[]
            fe_s[v]=[]
            fp_c[v]=[]
            fe_c[v]=[]

    if sfit:
        fit_toy(Sconfig, i, param_file, [ampBz,ampBp], vm, sc="sfit", **kwargs)
    if cfit:
        gen_weigths_for_cfit("Bz", Sconfig.configs[0]) # use Sconfig only for access to data and phsp
        gen_weigths_for_cfit("Bp", Sconfig.configs[1])
        Cconfig = MultiConfig([f"toystudy/CtoyBz{base}.yml", f"toystudy/CtoyBp{base}.yml"], total_same=True, vm=vm)
        fit_toy(Cconfig, i, param_file, [ampBz,ampBp], vm, sc="cfit", **kwargs)
        

def fit_toy(config, i, param_file, amps, vm, sc="sfit", fitloop=1):
    print(f"### Start {sc} toy {i}")
    set_same_D1(config)
    config.set_params(param_file)
    initNLL = config.get_fcn()({})
    fit_result = config.fit(batch=150000, method="BFGS")
    if not fit_result.success:
        print(f"$$$$$ {sc} failed")
        return
    for ni in range(fitloop-1):
        config.set_params(param_file)
        config.reinit_params()
        fit_res = config.fit(batch=150000, method="BFGS")
        if fit_res.success and (fit_res.min_nll < fit_result.min_nll):
            fit_result = fit_res
            if fit_result.min_nll - fit_res.min_nll > 1e-6:
                print("$$$ New Min Found")
    config.set_params(fit_result.params)
    improveNLL = (initNLL - fit_result.min_nll).numpy()
    '''for amp in amps:
        amp.cached_fun = amp.decay_group.sum_amp # turn off use_tf_function
    vm.rp2xy_all()
    fit_result = config.fit(batch=150000, method="BFGS")
    if not fit_result.success:
        print(f"$$$$$ {sc} failed in xy")
        return'''
    fit_error = config.get_params_error(fit_result, batch=30000)
    fit_result.set_error(fit_error)

    if sc == "sfit":
        for v in vm.trainable_vars:
            fp_s[v].append(fit_result.params[v])
            fe_s[v].append(fit_result.error[v])
        impNLL_s.append(improveNLL)
        print(f"@@@@@{sc} values{i}\n{fp_s}")
        print(f"@@@@@{sc} errors{i}\n{fe_s}")
        print(f"@@@@@{sc} change in NLL{i}\n{impNLL_s}")
    elif sc == "cfit":
        for v in vm.trainable_vars:
            fp_c[v].append(fit_result.params[v])
            fe_c[v].append(fit_result.error[v])
        impNLL_c.append(improveNLL)
        print(f"@@@@@{sc} values{i}\n{fp_c}")
        print(f"@@@@@{sc} errors{i}\n{fe_c}")
        print(f"@@@@@{sc} change in NLL{i}\n{impNLL_c}")


if __name__ == "__main__":
    Ntoy = 100 # edit
    fp_s = {}
    fe_s = {}
    impNLL_s = []
    fp_c = {}
    fe_c = {}
    impNLL_c = []
    print(f"$$$$$ Start fit {Ntoy} toys")
    for i in range(Ntoy):
        # edit below
        param_pulls(i, base="Z0", param_file="../DDspi/save/baseZ0_c/final_params_xy.json", sfit=True, cfit=True, fitloop=1)

