# TRON: A Fast Solver for Trajectory Optimization with Non-smooth Cost Functions

If you are using this code, please cite our work using the following BIBTEX citation:

```
bibtex citation here
```

## Install Dependencies
```bash
pip install -r requirements.txt

```
## Create directories

``` bash
./scripts/make_dirs.sh
```
## How to Run Experiments

### Lasso Problem with Synthetic Data

``` bash
python -m TRON.main_lasso
```

### Collision-Free Motion Planning for a Mobile Robot

```bash
python -m TRON.main --exp
```

### Sparse Optimal Control for a Surgical Steerable Needle

``` bash
python -m TRON.main_needle --exp
```

### Satellite Rendezvous Problem

``` bash
python -m TRON.main_satellite --exp
```

## Contributors

The repository is maintained and developed by [Anirudh Vemula](https://vvanirudh.github.io/) from the Search Based Planning Laboratory (SBPL), and Learning and Artificial Intelligence Laboratory (LairLab) at CMU
