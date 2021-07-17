from toy_fit import *
from tf_pwa.variable import VarsManager
from tf_pwa.applications import fit_fractions


def param_pulls(i, base="", param_file=None, sfit=True, cfit=True, **kwargs):
    if param_file is None:
        param_file = f"toystudy/params/base{base}_s.json"
    vm = VarsManager()
    vm.rp2xy_all() # fit in xy
    Sconfig = MultiConfig([f"toystudy/StoyBz{base}.yml", f"toystudy/StoyBp{base}.yml"], total_same=True, vm=vm)
    Sconfig.set_params(param_file)
    ampBz, ampBp = Sconfig.get_amplitudes()
    gen_toyBz(ampBz, Sconfig.configs[0])
    gen_toyBp(ampBp, Sconfig.configs[1])
    if i == 0: # initialization
        for v in vm.trainable_vars:
            fp_s[v]=[]
            fe_s[v]=[]
            fp_c[v]=[]
            fe_c[v]=[]
        for j, amp in enumerate([ampBz, ampBp]):
            for reson in amp.res:
                ffp_s[j][reson.name] = []
                ffe_s[j][reson.name] = []
                ffp_c[j][reson.name] = []
                ffe_c[j][reson.name] = []
        print("$$$", ffp_s)
        print("$$$", ffe_c)

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
            if fit_result.min_nll - fit_res.min_nll > 1e-6:
                print("$$$ New Min Found")
            fit_result = fit_res
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
        fp = fp_s
        fe = fe_s
        impNLL = impNLL_s
        ffp = ffp_s
        ffe = ffe_s
    elif sc == "cfit":
        fp = fp_c
        fe = fe_c
        impNLL = impNLL_c
        ffp = ffp_c
        ffe = ffe_c
    for v in vm.trainable_vars:
        fp[v].append(fit_result.params[v])
        fe[v].append(fit_result.error[v])
    impNLL.append(improveNLL)
    print(f"@@@@@{sc} values{i}\n{fp}")
    print(f"@@@@@{sc} errors{i}\n{fe}")
    print(f"@@@@@{sc} change in NLL{i}\n{impNLL}")
    cal_fit_frac(config, fit_result, amps, i, ffp, ffe)
    print(f"@###@{sc} Bz fit-frac values{i}\n{ffp[0]}")
    print(f"@###@{sc} Bz fit-frac errors{i}\n{ffe[0]}")
    print(f"@###@{sc} Bp fit-frac values{i}\n{ffp[1]}")
    print(f"@###@{sc} Bp fit-frac errors{i}\n{ffe[1]}")

def cal_fit_frac(config, fit_result, amps, i, ffp, ffe):
    for it, config_i in enumerate(config.configs):
        mcdata = config_i.get_phsp_noeff()
        fit_frac, err_frac = fit_fractions(
            config_i.get_amplitude(),
            mcdata,
            config.inv_he,
            fit_result.params,
        )
        for reson in amps[it].res:
            ffp[it][reson.name].append(fit_frac[reson.name])
            ffe[it][reson.name].append(err_frac[reson.name])


if __name__ == "__main__":
    Ntoy = 200 # edit
    fp_s = {}
    fe_s = {}
    ffp_s = [{}, {}] # fitfraction [Bz, Bp]
    ffe_s = [{}, {}]
    impNLL_s = []
    fp_c = {}
    fe_c = {}
    ffp_c = [{}, {}]
    ffe_c = [{}, {}]
    impNLL_c = []
    print(f"$$$$$ Start fit {Ntoy} toys")
    for i in range(Ntoy):
        # edit below
        param_pulls(i, base="MD", param_file="../DDspi/savec/MDZ0_c/final_params_xy.json", sfit=True, cfit=True, fitloop=1)

