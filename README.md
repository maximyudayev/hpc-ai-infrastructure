# HPC AI Infrastructure
Training, testing, and benchmarking PyTorch infrastructure for R&D workflows at eMedia Labs, KU Leuven. Uses multiple optimization and parallelization tricks, compatible with distributed multi-GPU setups, to process large out-of-core datasets.

Contains optimized distributed training routines for research areas targeted by eMedia's PhDs, with automation for deployment on the [Vlaams Supercomputer Centrum (VSC)](https://www.vscentrum.be/) via SLURM.

Classes are decoupled for modularity and readability, with separation of concern dictated by the user's JSON configuration files or CLI arguments.

## Derived Publications
> **Realtime ST-GCN: Adapting for Inference at the Edge**, Maxim Yudayev, Benjamin Filtjens and Josep Balasch, TNNLS 2023. [[Arxiv Preprint]](https://www.youtube.com/watch?v=BBJa32lCaaY)

## Supported Applications
- [X] Human activity recognition on long, untrimmed, multi-action sequences.

## TODO
- [ ] Add automated environment setup scripts (local and VSC).
- [ ] Explain where user files should go when using the project.

## Future Directions
- [ ] Convert scripts into microservice cloud endpoints (running on the VSC) to train user models on proprietary benchmark datasets (paid or with per-user quota).

## Installation
### Environment
Local environment uses Conda for ease and convenience. 

High Performance Computing for heavy-duty training and testing is done at Vlaams Supercomputer Centrum [(VSC)](https://www.vscentrum.be/), a Linux environment supercomputer for industry and academia in Flanders (Belgium).

#### **Local**
Create a Conda environment with all the dependencies, and clone the repository.
```shell
conda create -n rt-st-gcn --file requirements.txt
conda activate rt-st-gcn
git clone https://github.com/maximyudayev/hpc-ai-infrastructure.git
```

#### **HPC**
To speed-up training, in this project we used the high-performance computing infrastructure available to us by the [Vlaams Supercomputer Centrum (VSC)](https://www.vscentrum.be/).

Create a Conda environment identical to [Local setup](#local) or leverage optimized modules compiled with toolchains for the specific VSC hardware (Intel, FOSS): to do that, launch appropriate SLURM job scripts that load PyTorch using the `module` system instead of activating the Conda environments.

## Use
Edit `./main.py` to interface the processing infrastructure of the desired application.

## Citation
Please cite the following paper if you use this repository in your reseach.
```
@inproceedings{ ,
  title     = { },
  author    = { },
  booktitle = { },
  year      = { },
}
```

## Contact
For any questions, feel free to contact
```
Maxim Yudayev : maxim.yudayev@kuleuven.be
```

## Acknowledgements
The resources and services used in this work were provided by the VSC [(Flemish Supercomputer Center)](https://www.vscentrum.be/), funded by the Research Foundation - Flanders (FWO) and the Flemish Government.
