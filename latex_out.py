# methods to generate LaTeX source codes for parameter table and fit-fraction table
def error_print(x, err=None):
    import math
    if err is None:
        return ("{}").format(x)
    if err <= 0 or math.isnan(err):
        return ("{} ? {}").format(x, err)
    d = math.ceil(math.log10(err))
    b = 10 ** d
    b_err = err / b
    b_val = x / b
    if b_err < 0.355:  # 0.100 ~ 0.354
        dig = 2
    elif b_err < 0.950:  # 0.355 ~ 0.949
        dig = 1
    else:  # 0.950 ~ 0.999
        dig = 0
    err = round(b_err, dig) * b
    x = round(b_val, dig) * b
    d_p = dig - d
    if d_p > 0:
        return ("{0:.%df}±{1:.%df}" % (d_p, d_p)).format(x, err)
    return ("{0:.0f}±{1:.0f}").format(x, err)

class latex_out(object):
    def __init__(self, cand_dic, param_dic):
        self.cand_dic = cand_dic
        self.param_dic = param_dic

    def _get_point(self, s):
        import re
        partten = re.compile(r"([^\s]+)\s+([+-.e1234567890]+)\s+")
        ret = {}
        intersum = 0
        for i in s.split("\n"):
            g = partten.match(i)
            if g:
                name, fracv, _, error = i.split(" ")
                name = name.split("x")
                fracv = float(fracv)*100; error = float(error)*100
                fracv, error = error_print(fracv, error).split("±")
                fracv = float(fracv); error = float(error)
                if fracv>=0.001:
                    frac =f"${fracv}\pm{error}$"
                else:
                    frac = "0.0"
                if len(name) == 1:
                    l, r = name*2
                    intersum += fracv
                elif len(name) == 2:
                    l, r = name
                else:
                    raise Exception("error {}".format(name))
                if l not in ret:
                    ret[l] = {}
                ret[l][r] = frac
        return ret, intersum
    def _get_table(self, s):
        idx = list(s)
        n_idx = len(idx)
        idx_map = dict(zip(idx, range(n_idx)))
        ret = []
        for i in range(n_idx):
            ret.append([0.0 for j in range(n_idx)])
        for i, k in s.items():
            for j, v in k.items():
                ret[idx_map[i]][idx_map[j]] = v
        return idx, ret
    def _frac_table(self, frac_txt, cand_dic):
        s, intersum = self._get_point(frac_txt)
        idx, table = self._get_table(s)
        ret = []
        for i, k in enumerate(table):
            tmp = []
            for j, v in enumerate(k):
                if i < j:
                    tmp.append("-")
                else:
                    tmp.append("{}".format(v))
            ret.append(tmp)
        lenk = len(ret[0])
        ret_string = ""
        for i, k in zip(idx, ret):
            ret_string += (cand_dic[i]+"\t&")
            for v, kk in zip(k, range(lenk)):
                if kk == lenk-1:
                    ret_string += (v+"\\\\")
                else:
                    ret_string += (v+"\t&")
            ret_string += "\n"
        return ret_string, intersum, lenk
    def fitfrac_table(self, tfpwa_string, cand_dic=None):
        if cand_dic == None:
            cand_dic = self.cand_dic
        ret_string, intersum, lenk = self._frac_table(tfpwa_string, cand_dic)
        head = r"""\begin{table}[htp]
\caption{Fit-fraction interference table. (All terms smaller than 0.1\% are given as 0 without uncertainty.)}
\begin{center}\resizebox{\textwidth}{!}{\begin{tabular}{ | c | """
        for i in range(lenk):
            head += "c "
        head += "| }\n\hline\n"
        tail =r"""\hline
\multicolumn{"""+str(lenk+1)+r"""}{| l |}{Sum of fit-fractions: """+f"{intersum:.1f}"+r""" (units: \%)}\\
\hline
\end{tabular}}\end{center}
\label{tab:interference_table}
\end{table}"""
        outstring = head+ret_string+tail
        return outstring
    
    
    def _process_input_param_string(self, inpstr):
        line = inpstr.split("\n")
        self.param_var_dic = {}
        for l in line:
            ll = l.split(" ")
            if len(ll) == 2:
                self.param_var_dic[ll[0]] = str(ll[1])
            elif len(ll) == 4:
                self.param_var_dic[ll[0]] = f"${ll[1]}\pm{ll[3]}$"
    def _traverse_dic(self, dic, layer=1, string=None):
        if string is None:
            string = ""
        for i in dic:
            if i == "Amplitude":
                tstr = string
            else:
                tstr = "\t"
            if layer == 1:
                tstr += f"\\hline\n{i}"
            else:
                #if layer == 2:
                tstr += "\t& "
                tstr += f"{i}\t& "
            if type(dic[i]) is dict:
                self._traverse_dic(dic[i], layer+1, tstr)
            else:
                if dic[i] in self.param_var_dic:
                    if i == "Mass" or i == "Width":
                        if r"\pm" not in self.param_var_dic[dic[i]]:
                            continue # skip fixed PDG mass and width
                    tstr += (self.param_var_dic[dic[i]]+"\\\\\n")
                    self.param_latex_table += tstr
    def param_table(self, inpstr, param_dic=None):
        if param_dic is None:
            param_dic = self.param_dic
        self._process_input_param_string(inpstr)
        self.param_latex_table = ""
        self._traverse_dic(param_dic)
        outstring = r"""\begin{center}\begin{longtable}{ c c c }
  \caption{Results of fit parameters.}
  \label{tab:param_app}\\
    \hline\hline\hline
    \multicolumn{2}{c}{Parameter} & Fit result \\
    \hline
    \endfirsthead
    
    \multicolumn{3}{c}%
{{\tablename\ \thetable{} (continued)}} \\
    \hline\hline
    \multicolumn{2}{c}{Parameter} & Fit result \\
    \hline
    \endhead
    
    \hline \multicolumn{3}{r}{{Continued next page}} \\ \hline\hline
    \endfoot
    
    \hline\hline\hline
    \endlastfoot
    """
        outstring += self.param_latex_table
        outstring += r"\end{longtable}\end{center}"
        return outstring


cand_dic = {
    "D1_2007": "$D^*(2007)^0$",
    "D1_2010": "$D^*(2010)^-$",
    "D0_2300": "$D_0^*(2300)$",
    "D2_2460": "$D_2^*(2460)$",
    "D1_2600": "$D_1^*(2600)$",
    "D3_2750": "$D_3^*(2750)$",
    "D1_2760": "$D_1^*(2760)$", # only seen neutral
    "DJ_2640": "$D_J^*(2640)$", # only seen charged
    "DJ_3000": "$D_J^*(3000)$", # only seen neutral
    "D0_2900": "$D_0^*(2900)$", # theoretical prediction
    "DPi_S": r"spline($D\pi$)S",
    "NR_DPi0": r"NR($D\pi$)S",
    "Z0": "$Z_0$",
    "Z1": "$Z_1$",
    "Z2": "$Z_2$",
    "Z3": "$Z_3$",
    "Dstar": r"$D\pi$",
    "DsPi": r"$D_s\pi$",
    "DDs": "$DD_s$",
}
assist_dic = {
    "$D^*(2007)^0$": ["B->D1_2007.DsD1_2007->D.Pi_total_0", "D1_2007"],
    "$D^*(2010)^-$": ["B->D1_2010.DsD1_2010->D.Pi_total_0", "D1_2010"],
    "$D_0^*(2300)$": ["B->D0_2300.DsD0_2300->D.Pi_total_0", "D0_2300"],
    "$D_2^*(2460)$": ["B->D2_2460.DsD2_2460->D.Pi_total_0", "D2_2460"],
    "$D_1^*(2600)$": ["B->D1_2600.DsD1_2600->D.Pi_total_0", "D1_2600"],
    "$D_3^*(2750)$": ["B->D3_2750.DsD3_2750->D.Pi_total_0", "D3_2750"],
    "$D_1^*(2760)$": ["B->D1_2760.DsD1_2760->D.Pi_total_0", "D1_2760"],
    "$D_J^*(2640)$": ["B->DJ_2640.DsDJ_2640->D.Pi_total_0", "DJ_2640"],
    "$D_J^*(3000)$": ["B->DJ_3000.DsDJ_3000->D.Pi_total_0", "DJ_3000"],
    "$D_0^*(2900)$": ["B->D0_2900.DsD0_2900->D.Pi_total_0", "D0_2900"],
    r"spline($D\pi$)S": ["B->DPi_S.DsDPi_S->D.Pi_total_0", "DPi_S_point"],
    r"NR($D\pi$)S": ["B->NR_DPi0.DsNR_DPi0->D.Pi_total_0", "NR_DPi0"],
    "$Z_0$": ["B->Z0.DZ0->Ds.Pi_total_0", "Z0"],
    "$Z_1$": ["B->Z1.DZ1->Ds.Pi_total_0", "Z1"],
    "$Z_2$": ["B->Z2.DZ2->Ds.Pi_total_0", "Z2"],
    "$Z_3$": ["B->Z3.DZ3->Ds.Pi_total_0", "Z3"],
}
param_dic = {}
for i in assist_dic:
    param_dic[i] = {"Amplitude": assist_dic[i][0]+"r", "Phase": assist_dic[i][0]+"i"}
    if i[0] == "$":
        param_dic[i]["Mass"] = assist_dic[i][1]+"_mass"
        param_dic[i]["Width"] = assist_dic[i][1]+"_width"
    elif i == r"NR($D\pi$)S":
        param_dic[i][r"Exponential $\alpha$"] = assist_dic[i][1]+"_a"
    elif i == r"spline($D\pi$)S":
        for j in range(12):
            param_dic[i][f"Spline point {j} x"] = assist_dic[i][1]+f"_{j}r"
            param_dic[i][f"Spline point {j} y"] = assist_dic[i][1]+f"_{j}i"

ltx = latex_out(cand_dic, param_dic)





# example strings
instr = """
D1_2010_mass 2.01026
D1_2010_width 8.34e-05
B->D1_2010.DsD1_2010->D.Pi_total_0r 2.69 +/- 0.11
B->D1_2010.DsD1_2010->D.Pi_total_0i 3.00 +/- 0.06
B->D1_2010.Ds_g_ls_0r 1.0
B->D1_2010.Ds_g_ls_0i 0.0
D1_2010->D.Pi_g_ls_0r 1.0
D1_2010->D.Pi_g_ls_0i 0.0
D2_2460_mass 2.4611
D2_2460_width 0.0473
D1_2600_mass 2.627
D1_2600_width 0.141
D1_2007_mass 2.00685
D1_2007_width 0.0001
Z0_mass 2.940 +/- 0.013
Z0_width 0.133 +/- 0.028
DPi_S_point_0r -0.12 +/- 0.24
DPi_S_point_0i -1.58 +/- 0.21
DPi_S_point_1r 0.31 +/- 0.10
DPi_S_point_1i -1.35 +/- 0.07
DPi_S_point_2r 0.76 +/- 0.07
DPi_S_point_2i -0.94 +/- 0.07
DPi_S_point_3r 1.00 +/- 0.05
DPi_S_point_3i -0.62 +/- 0.07
DPi_S_point_4r 1.0
DPi_S_point_4i 0.0
DPi_S_point_5r 0.508 +/- 0.033
DPi_S_point_5i 0.26 +/- 0.04
DPi_S_point_6r 0.411 +/- 0.034
DPi_S_point_6i 0.265 +/- 0.030
DPi_S_point_7r 0.09 +/- 0.05
DPi_S_point_7i 0.427 +/- 0.027
DPi_S_point_8r -0.03 +/- 0.04
DPi_S_point_8i 0.302 +/- 0.024
DPi_S_point_9r -0.161 +/- 0.025
DPi_S_point_9i 0.069 +/- 0.035
DPi_S_point_10r -0.05 +/- 0.05
DPi_S_point_10i 0.13 +/- 0.04
B->D2_2460.DsD2_2460->D.Pi_total_0r 1.0
B->D2_2460.DsD2_2460->D.Pi_total_0i 0.0
B->D2_2460.Ds_g_ls_0r 1.0
B->D2_2460.Ds_g_ls_0i 0.0
D2_2460->D.Pi_g_ls_0r 1.0
D2_2460->D.Pi_g_ls_0i 0.0
B->D1_2600.DsD1_2600->D.Pi_total_0r 0.368 +/- 0.024
B->D1_2600.DsD1_2600->D.Pi_total_0i 0.23 +/- 0.07
B->D1_2600.Ds_g_ls_0r 1.0
B->D1_2600.Ds_g_ls_0i 0.0
D1_2600->D.Pi_g_ls_0r 1.0
D1_2600->D.Pi_g_ls_0i 0.0
B->D1_2007.DsD1_2007->D.Pi_total_0r 2.689391865616644
B->D1_2007.DsD1_2007->D.Pi_total_0i 3.0048519324809457
B->D1_2007.Ds_g_ls_0r 1.0
B->D1_2007.Ds_g_ls_0i 0.0
D1_2007->D.Pi_g_ls_0r 1.0
D1_2007->D.Pi_g_ls_0i 0.0
B->Z0.DZ0->Ds.Pi_total_0r -0.120 +/- 0.028
B->Z0.DZ0->Ds.Pi_total_0i 2.74 +/- 0.21
B->Z0.D_g_ls_0r 1.0
B->Z0.D_g_ls_0i 0.0
Z0->Ds.Pi_g_ls_0r 1.0
Z0->Ds.Pi_g_ls_0i 0.0
B->DPi_S.DsDPi_S->D.Pi_total_0r 1.19 +/- 0.04
B->DPi_S.DsDPi_S->D.Pi_total_0i -1.07 +/- 0.04
B->DPi_S.Ds_g_ls_0r 1.0
B->DPi_S.Ds_g_ls_0i 0.0
DPi_S->D.Pi_g_ls_0r 1.0
DPi_S->D.Pi_g_ls_0i 0.0
"""
tfpwa_string = """
D2_2460 0.248 +/- 0.007
D1_2600 0.041 +/- 0.005
D1_2600xD2_2460 0.00074 +/- 0.00012
D1_2010 0.147 +/- 0.011
D1_2010xD1_2600 -0.003 +/- 0.005
D1_2010xD2_2460 -0.00083 +/- 0.00012
Z0 0.017 +/- 0.005
Z0xD1_2010 -0.015 +/- 0.006
Z0xD1_2600 0.0107 +/- 0.0021
Z0xD2_2460 -0.0014 +/- 0.0035
DPi_S 0.563 +/- 0.016
DPi_SxZ0 -0.006 +/- 0.011
DPi_SxD1_2010 -0.00171 +/- 0.00019
DPi_SxD1_2600 -0.00026 +/- 0.00008
DPi_SxD2_2460 0.00002 +/- 0.00007
"""
if __name__ == "__main__":
    ffltx=ltx.fitfrac_table(tfpwa_string)
    parltx=ltx.param_table(instr)
    print(ffltx)
    print("\n\n")
    print(parltx)