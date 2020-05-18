# [TRON: A Fast Solver for Trajectory Optimization with Non-smooth Cost Functions](https://arxiv.org/abs/2003.14393)

If you are using this code, please cite our work using the following BIBTEX citation:

```
@article{DBLP:journals/corr/abs-2003-14393,
  author    = {Anirudh Vemula and
               J. Andrew Bagnell},
  title     = {{TRON:} {A} Fast Solver for Trajectory Optimization with Non-Smooth
               Cost Functions},
  journal   = {CoRR},
  volume    = {abs/2003.14393},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.14393},
  archivePrefix = {arXiv},
  eprint    = {2003.14393},
  timestamp = {Thu, 02 Apr 2020 15:34:08 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2003-14393.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
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
