# cross_validated_causal

## No-covariate setting
``mean_eps.py``: Varying bias.

``mean_n_obs.py``: Varying the number of observational data.
### Usage
Install missing packages if needed.

Run 
```
python mean_eps.py
```
or 
```
python mean_n_obs.py
```
Detailed usage see python scripts.

## Linear setting
``mean_eps.py``: Varying bias.

``mean_n_obs.py``: Varying the number of observational data.
### Usage
Choice 1: Directly run 
```
python linear_eps.py
``` 
or 
```
python linear_n_obs.py
``` 

Choise 2: Use a bash script and specify ``--cpus-per-task`` in the  for parallel computing.

Detailed usage see python scripts.

## Experiments on the LaLonde dataset
### Data
Download the ``.txt`` files of **NSW Data Files (Dehejia-Wahha Sample)** and **PSID and CPS Data Files** from [link](http://users.nber.org/~rdehejia/nswdata2.html) and put them into a ``\data`` folder.
### Usage
``lalonde_baseline.Rmd``: Estimation and bootstrap for baselines in Table 2 and 3. 

``read_lalonde_data.R``: Read data. 

