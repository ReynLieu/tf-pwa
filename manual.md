### Download the package at https://github.com/ReynLieu/tf-pwa
```git clone https://github.com/ReynLieu/tf-pwa.git
```

### Then switch to branch DDspi
```mv tf-pwa DDspi
cd DDspi
git checkout DDspi
```

### To get the data files 
```mkdir ../dataDDspi
scp -r liuyr@login.lzuhep.lzu.edu.cn:/data2/liuyr/workspace/dataDDspi/dat* ../dataDDspi/ 
```
(Ask me for password)

### Then you need the configuration files
For sfit, `touch config.yml config_Bp.yml`.
Then open `config.sample.yml`, paste lines headed by `##### config.yml` (ended by `##### configC.yml`) into `config.yml`; paste lines headed by `##### config_Bp.yml` (ended by `##### configC_Bp.yml`) into `config_Bp.yml`

For cfit `cp config.yml configC.yml`, `cp config_Bp.yml configC_Bp.yml`.
Then subsitute the `data:` block in `configC.yml` with lines headed by `##### configC.yml` in `config.sample.yml`; subsitute the `data:` block in `configC_Bp.yml` with lines headed by `##### configC_Bp.yml` in `config.sample.yml`.

### Now you have everything in hand, and you can run the script for fitting!
By using some initial parameter files I left under `toystudy/params/`, you will immediately converge and see the outputs.

For simultaneous sfit of Bz, Bp channels
```python fit.py -c=config.yml,config_Bp.yml -i=toystudy/params/base_s.json > log_s.txt
```

For simultaneous cfit of Bz, Bp channels
```python fit.py -c=configC.yml,configC_Bp.yml -i=toystudy/params/base_C.json > log_c.txt
```

You shall see some generated files everytime you run `python fit.py`, including figures under `figure/`. To save these files before being overwritten, you can run `./state_cache.sh save/test1`, then the related output files will be copied under `save/test1/`.

### Just some basics. Enjoy~
