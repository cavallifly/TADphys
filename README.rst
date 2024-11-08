

+-----------------------+-+
|                       | |
| Current version: pipeline_v0.2.722  |
|                       | |
+-----------------------+-+


TADphys.

Documentation
*************

**Install LAMMPS as a shared library**
   | 1 - Download lammps
   | git clone -b stable https://github.com/lammps/lammps.git mylammps
   
   | 2 - Install lammps as a shared library
   | cd ../../src/
   | include "-DLAMMPS_EXCEPTIONS" in the LMP_INC line in src/MAKE/Makefile.mpi
   | make yes-molecule
   | make mpi mode=shlib
   | make install-python

   | cd ../../

**Install packages**
   | conda install -y scipy           # scientific computing in python
   | conda install -y numpy           # scientific computing in python
   | conda install -y matplotlib      # to produce plots
   | conda install -y -c https://conda.anaconda.org/bcbio pysam # to deal with SAM/BAM files

**Install TADphys**
   | 1 - Download TADphys from the Github repository
   | git clone https://github.com/MarcoDiS/TADphys.git -b TADphys TADphys

   | 2 - Install TADphys
   | cd TADphys
   | python setup.py install
   | cd ..

**Try TADdyn**
   | cd test/
   | python test_TADphys.py

Citation
********
Please, cite this article if you use TADphys.

Marco Di Stefano, Ralph Stadhouders, Irene Farabella, David Castillo, François Serra, Thomas Graf, Marc A. Marti-Renom.
**Dynamic simulations of transcriptional control during cell reprogramming reveal spatial chromatin caging.**
*bioRxiv* 642009; `doi: https://doi.org/10.1101/642009`_

Methods implemented in TADphys
-----------------------------
In the actual implementation, TADphys relies on TADbit for the models' analysis
and on LAMMPS [Plimpton]_ for the implementation of the simulations.

Bibliography
************

.. [Plimpton] Plimpton, S. Fast Parallel Algorithms for Short-Range Molecular Dynamics. J Comp Phys 117, 1-19 (1995) and Fiorin, G., Klein, M.L. & Hénin, J. Using collective variables to drive molecular dynamics simulations. Molecular Physics 111, 3345-3362 (2013).
