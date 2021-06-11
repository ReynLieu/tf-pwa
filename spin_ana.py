from toy_fit import sig_test


def spin_ana(Ntoy, null="Z0", fitloop=1, sfit=True, cfit=True, param_file=None):
    if null == "Z0":
        alternative = "Z1"
    elif null == "Z1":
        alternative = "Z0"
    if param_file is None:
        param_file = f"toystudy/params/base{null}_s.json"
    SdNLL = []
    CdNLL = []
    for i in range(Ntoy):
        print(f"##### Start toy {i}: null {null} alternative {alternative}")
        Sdnll, Cdnll = sig_test(sfit=sfit, cfit=cfit, null=null, alternative=alternative, param_file=param_file, fitloop=fitloop)
        print("$$$$$dnll:", Sdnll, Cdnll)
        SdNLL.append(Sdnll)
        CdNLL.append(Cdnll)
        print(f"deltaNLL for Sfit:\n{SdNLL}")
        print(f"deltaNLL for Cfit:\n{CdNLL}")
    return SdNLL, CdNLL

if __name__ == "__main__":
    SdNLL, CdNLL = spin_ana(100, null="Z0", fitloop=3, sfit=True, cfit=True) # edit
    print(f"########## deltaNLL for Sfit:\n{SdNLL}")
    print(f"########## deltaNLL for Cfit:\n{CdNLL}")