# TADphys
Scripts for the biophysical modelling simulations

**Documentation**
*************
**Download TADphys repository**

Download TADphys from the Github repository
```bash
git clone https://github.com/cavallifly/TADphys.git -b TADphys TADphys
```

**Create TADphys conda environment**

To use TADphys, we recommend installing the package manager conda from 'https://conda.io/projects/conda/en/latest/user-guide/install/index.html.'

Next, you can use the provided TADphys.yml to create the TADphys environment (Execution time ~30 minutes)
```bash
conda env create -f TADphys.yml
```

**Install LAMMPS as a shared library**
  
1 - Download lammps
```bash
git clone -b stable https://github.com/lammps/lammps.git mylammps
```

2 - Install lammps as a shared library
```bash
cd ../../src/
Open the file MAKE/Makefile.mpi with your preferred text editor and add "-DLAMMPS_EXCEPTIONS" in the LMP_INC line
make yes-molecule
make mpi mode=shlib
make install-python
cd ../../
```

**Install TADphys**
   
Install TADphys
```bash
cd TADphys
python setup.py install
cd ..
```

Methods implemented in TADphys
-----------------------------
In the actual implementation, TADphys relies on TADdyn [[1]](#1) and TADbit [[2]](#2) for the general structure of the package and LAMMPS [[3]](#3) for the implementation of the simulations.

**Citation**
********
Please, cite this article if you use TADphys.

Ivana Jerković, Marco Di Stefano, Hadrien Reboul, Michael F Szalay,  Davide Normanno, Giorgio L Papadopoulos, Frederic Bantignies, Giacomo Cavalli.
**A Scaffolding Element Rewires Local 3D Chromatin Architecture During Differentiation.**
*bioRxiv* 642009; [doi.org/10.1101/2024.05.23.595561](https://doi.org/10.1101/2024.05.23.595561)

**Bibliography**
************
<a id="1">[1]</a>
Di Stefano, M. et al. Transcriptional activation during cell reprogramming correlates with the formation of 3D open chromatin hubs. Nature communications 11 (1), 1-12 (2020).

<a id="2">[2]</a>
Serra, F. et al. Automatic analysis and 3D-modelling of Hi-C data using TADbit reveals structural features of the fly chromatin colors. PLoS Comp Biol 1005665 (2017).
	   
<a id="3">[3]</a>
Plimpton, S. Fast Parallel Algorithms for Short-Range Molecular Dynamics. J Comp Phys 117, 1-19 (1995).
