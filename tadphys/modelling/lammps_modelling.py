"""
16 Mar 2019


"""
from string import ascii_uppercase as uc, ascii_lowercase as lc
from os.path import exists
from random import uniform, randint, seed, random, sample, shuffle, choice
from pickle import load, dump
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from math import atan2
from itertools import combinations, product, chain
from shutil import copyfile
from operator import itemgetter

import sys
import copy
import os
import shutil
import multiprocessing
import time

from numpy import sin, cos, arccos, sqrt, fabs, pi, zeros, log, exp, array_equal, full_like, ones
from scipy import spatial
import numpy as np
from mpi4py import MPI
from lammps import lammps

from tadphys.modelling import LAMMPS_CONFIG as CONFIG
from tadphys.modelling.lammpsmodel import LAMMPSmodel
from tadphys.modelling.restraints import HiCBasedRestraints

class InitalConformationError(Exception):
    """
    Exception to handle failed initial conformation.
    """
    pass

def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    failedSeedLog = kwargs.get('failedSeedLog', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        print("Model took more than %s seconds to complete ... canceling" % str(timeout))
        p.terminate()
        raise
    except:
        print("Unknown error with process")
        if failedSeedLog != None:
            failedSeedLog, k = failedSeedLog
            with open(failedSeedLog, 'a') as f:
                f.write('%s\t%s\n' %(k, 'Failed'))
        p.terminate()
        raise

def generate_lammps_models(zscores, resolution, nloci, start=1, n_models=5000,
                       n_keep=1000, close_bins=1, n_cpus=1,
                       verbose=0, outfile=None, config=None,
                       values=None, coords=None, zeros=None,
                       first=None, container=None,tmp_folder=None,timeout_job=10800,
                       initial_conformation=None, connectivity={1:["FENE", 30.0, 1.5, 1.0, 1.0]},
                       timesteps_per_k=10000,keep_restart_out_dir=None,
                       kfactor=1, adaptation_step=False, cleanup=False,
                       hide_log=True, remove_rstrn=[], initial_seed=0,
                       restart_path=False, store_n_steps=10,
                       useColvars=False):
    """
    This function generates three-dimensional models starting from Hi-C data.
    The final analysis will be performed on the n_keep top models.

    :param zscores: the dictionary of the Z-score values calculated from the
       Hi-C pairwise interactions
    :param resolution:  number of nucleotides per Hi-C bin. This will be the
       number of nucleotides in each model's particle
    :param nloci: number of particles to model (may not all be present in
       zscores)
    :param None coords: a dictionary like:
       ::

         {'crm'  : '19',
          'start': 14637,
          'end'  : 15689}

    :param 5000 n_models: number of models to generate
    :param 1000 n_keep: number of models used in the final analysis (usually
       the top 20% of the generated models). The models are ranked according to
       their objective function value (the lower the better)
    :param 1 close_bins: number of particles away (i.e. the bin number
       difference) a particle pair must be in order to be considered as
       neighbors (e.g. 1 means consecutive particles)
    :param n_cpus: number of CPUs to use
    :param False verbose: if set to True, information about the distance, force
       and Z-score between particles will be printed. If verbose is 0.5 than
       constraints will be printed only for the first model calculated.
    :param None values: the normalized Hi-C data in a list of lists (equivalent
       to a square matrix)
    :param None config: a dictionary containing the standard
       parameters used to generate the models. The dictionary should contain
       the keys kforce, lowrdist, maxdist, upfreq and lowfreq. Examples can be
       seen by doing:

       ::

         from tadphys.modelling.HIC_CONFIG import HIC_CONFIG

         where CONFIG is a dictionary of dictionaries to be passed to this function:

       ::

         CONFIG = {
          'dmel_01': {
              # Paramaters for the Hi-C dataset from:
              'reference' : 'victor corces dataset 2013',

              # Force applied to the restraints inferred to neighbor particles
              'kforce'    : 5,

              # Space occupied by a nucleotide (nm)
              'scale'     : 0.005

              # Strength of the bending interaction
              'kbending'     : 0.0, # OPTIMIZATION:

              # Maximum experimental contact distance
              'maxdist'   : 600, # OPTIMIZATION: 500-1200

              # Minimum thresholds used to decide which experimental values have to be
              # included in the computation of restraints. Z-score values bigger than upfreq
              # and less that lowfreq will be include, whereas all the others will be rejected
              'lowfreq'   : -0.7 # OPTIMIZATION: min/max Z-score

              # Maximum threshold used to decide which experimental values have to be
              # included in the computation of restraints. Z-score values greater than upfreq
              # and less than lowfreq will be included, while all the others will be rejected
              'upfreq'    : 0.3 # OPTIMIZATION: min/max Z-score

              }
          }
    :param None first: particle number at which model should start
    :param None container: restrains particle to be within a given object. Can
       only be a 'cylinder', which is, in fact a cylinder of a given height to
       which are added hemispherical ends. This cylinder is defined by a radius,
       its height (with a height of 0 the cylinder becomes a sphere) and the
       force applied to the restraint. E.g. for modeling E. coli genome (2
       micrometers length and 0.5 micrometer of width), these values could be
       used: ['cylinder', 250, 1500, 50], and for a typical mammalian nuclei
       (6 micrometers diameter): ['cylinder', 3000, 0, 50]
    :param None tmp_folder: path to a temporary file created during
        the clustering computation. Default will be created in /tmp/ folder
    :param 10800 timeout_job: maximum seconds a job can run in the multiprocessing
        of lammps before is killed
    :param initial_conformation: lammps input data file with the particles initial conformation.
    :param True hide_log: do not generate lammps log information
    :param FENE connectivity: use FENE for a fene bond or harmonic for harmonic
        potential for neighbours
    :param None keep_restart_out_dir: path to write files to restore LAMMPs
                session (binary)
    :param True cleanup: delete lammps folder after completion
    :param [] remove_rstrn: list of particles which must not have restrains
    :param 0 initial_seed: Initial random seed for modelling.
    :param False restart_path: path to files to restore LAMMPs session (binary)
    :param 10 store_n_steps: Integer with number of steps to be saved if 
        restart_file != False
    :param False useColvars: True if you want the restrains to be loaded by colvars

    :returns: a Tadphys models dictionary
    """

    if not tmp_folder:
        tmp_folder = '/tmp/tadphys_tmp_%s/' % (
            ''.join([(uc + lc)[int(random() * 52)] for _ in range(4)]))
    else:
        if tmp_folder[-1] != '/':
            tmp_folder += '/'
        randk = ''.join([(uc + lc)[int(random() * 52)] for _ in range(4)])
        tmp_folder = '%s%s/' %(tmp_folder, randk)
    while os.path.exists(tmp_folder):
        randk = ''.join([(uc + lc)[int(random() * 52)] for _ in range(4)])
        tmp_folder = '%s%s/' %(tmp_folder[:-1], randk)
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # Setup CONFIG
    if isinstance(config, dict):
        CONFIG.HiC.update(config)
    elif config:
        raise Exception('ERROR: "config" must be a dictionary')

    global RADIUS

    #RADIUS = float(resolution * CONFIG['scale']) / 2
    RADIUS = 0.5
    CONFIG.HiC['resolution'] = resolution
    CONFIG.HiC['maxdist'] = CONFIG.HiC['maxdist'] / (float(resolution * CONFIG.HiC['scale']))

    global LOCI
    if first is None:
        first = min([int(j) for i in zscores[0] for j in zscores[0][i]] +
                    [int(i) for i in zscores[0]])
    LOCI = list(range(first, nloci + first))
    LOCI = 20000 
    
    # random inital number
    global START
    START = start
    # verbose
    global VERBOSE
    VERBOSE = verbose
    #VERBOSE = 3

    HiCRestraints = [HiCBasedRestraints(nloci,RADIUS,CONFIG.HiC,resolution, zs,
                 chromosomes=coords, close_bins=close_bins,first=first,
                 remove_rstrn=remove_rstrn) for zs in zscores]

    run_time = 1000
    
    colvars = 'colvars.dat'

    steering_pairs = None
    time_dependent_steering_pairs = None
    if len(HiCRestraints) > 1:
        time_dependent_steering_pairs = {
            'colvar_input'              : HiCRestraints,
            'colvar_output'             : colvars,
            'chrlength'                 : nloci,
            'binsize'                   : resolution,
            'timesteps_per_k_change'    : [float(timesteps_per_k)]*6,
            'k_factor'                  : kfactor,
            'perc_enfor_contacts'       : 100.,
            'colvar_dump_freq'          : int(timesteps_per_k/100),
            'adaptation_step'           : adaptation_step,
        }
        if not initial_conformation:
            initial_conformation = 'random'
    else:
        steering_pairs = {
            'colvar_input': HiCRestraints[0],
            'colvar_output': colvars,
            'binsize': resolution,
            'timesteps_per_k'           : timesteps_per_k,
            'k_factor'                  : kfactor,
            'colvar_dump_freq'          : int(timesteps_per_k/100),
            'timesteps_relaxation'      : int(timesteps_per_k*10)
        }
        if not initial_conformation:
            initial_conformation = 'random'

    if not container:
        container = ['cube',1000.0] # http://lammps.sandia.gov/threads/msg48683.html

    ini_sm_model = None
    sm_diameter = 1
    if initial_conformation != 'random':
        if isinstance(initial_conformation, dict):
            sm = [initial_conformation]
            sm[0]['x'] = sm[0]['x'][0:nloci]
            sm[0]['y'] = sm[0]['y'][0:nloci]
            sm[0]['z'] = sm[0]['z'][0:nloci]
        sm_diameter = float(resolution * CONFIG.HiC['scale'])
        for single_m in sm:
            for i in range(len(single_m['x'])):
                single_m['x'][i] /= sm_diameter
                single_m['y'][i] /= sm_diameter
                single_m['z'][i] /= sm_diameter
            cm0 = single_m.center_of_mass()
            for i in range(len(single_m['x'])):
                single_m['x'][i] -= cm0['x']
                single_m['y'][i] -= cm0['y']
                single_m['z'][i] -= cm0['z']
        ini_sm_model = [[single_sm.copy()] for single_sm in sm]

    models, ini_model = lammps_simulate(lammps_folder=tmp_folder,
                                        run_time=run_time,
                                        initial_conformation=ini_sm_model,
                                        connectivity=connectivity,
                                        steering_pairs=steering_pairs,
                                        time_dependent_steering_pairs=time_dependent_steering_pairs,
                                        initial_seed=initial_seed,
                                        n_models=n_keep,
                                        n_keep=n_keep,
                                        n_cpus=n_cpus,
                                        keep_restart_out_dir=keep_restart_out_dir,
                                        confining_environment=container,
                                        timeout_job=timeout_job,
                                        cleanup=cleanup, to_dump=int(timesteps_per_k/100.),
                                        hide_log=hide_log,
                                        chromosome_particle_numbers=chromosome_particle_numbers,
                                        restart_path=restart_path,
                                        store_n_steps=store_n_steps,
                                        useColvars=useColvars)
#     for i, m in enumerate(models.values()):
#         m['index'] = i
    if outfile:
        if exists(outfile):
            old_models, _ = load(open(outfile))
        else:
            old_models, _ = {}, {}
        models.update(old_models)
        out = open(outfile, 'w')
        dump((models), out)
        out.close()
    else:
        stages = {}
        trajectories = {}
        timepoints = None
        if len(HiCRestraints)>1:
            timepoints = time_dependent_steering_pairs['colvar_dump_freq']
            nbr_produced_models = len(models)//(timepoints*(len(HiCRestraints)-1))
            stages[0] = [i for i in range(nbr_produced_models)]

            for sm_id, single_m in enumerate(ini_model):
                for i in range(len(single_m['x'])):
                    single_m['x'][i] *= sm_diameter
                    single_m['y'][i] *= sm_diameter
                    single_m['z'][i] *= sm_diameter
                
                lammps_model = LAMMPSmodel({ 'x'          : single_m['x'],
                                              'y'          : single_m['y'],
                                              'z'          : single_m['z'],
                                              'index'      : sm_id+1+initial_seed,
                                              'cluster'    : 'Singleton',
                                              'objfun'     : single_m['objfun'] if 'objfun' in single_m else 0,
                                              'log_objfun' : single_m['log_objfun'] if 'log_objfun' in single_m else [],
                                              'radius'     : float(CONFIG.HiC['resolution'] * \
                                                                   CONFIG.HiC['scale'])/2,
                                              'rand_init'  : str(sm_id+1+initial_seed)})

                models[sm_id] = lammps_model
            for timepoint in range((len(HiCRestraints)-1)*timepoints):
                stages[timepoint+1] = [(t+nbr_produced_models+timepoint*nbr_produced_models)
                                       for t in range(nbr_produced_models)]
            for traj in range(nbr_produced_models):
                trajectories[traj] = [stages[t][traj] for t in range(timepoints+1)]

        model_ensemble = {
            'loci':             len(LOCI), 
            'models':           models, 
            'resolution':       resolution,
            'original_data':    values if len(HiCRestraints)>1 else values[0],
            'zscores':          zscores,
            'config':           CONFIG.HiC, 
            'zeros':            zeros,
            'restraints':       HiCRestraints[0]._get_restraints(),
            'stages':           stages,
            'trajectories':     trajectories,
            'models_per_step':  timepoints
        }

        return model_ensemble
# Initialize the lammps simulation with standard polymer physics based
# interactions: chain connectivity (FENE) ; excluded volume (WLC) ; and
# bending rigidity (KP)
def init_lammps_run(lmp, initial_conformation,
                    neighbor=CONFIG.neighbor,
                    hide_log=True,
                    timestep=CONFIG.timestep,
                    reset_timestep=0,
                    chromosome_particle_numbers=None,
                    connectivity   = {1:["FENE", 30.0, 1.5, 1.0, 1.0]},
                    excludedVolume = {(1,1):["LJ"  ,  1.0, 1.0, 1.12246152962189]},
                    nveLimit = False,
                    persistence_length =  0.0,
                    restart_file=False):

    """
    Initialise the parameters for the computation in lammps job

    :param lmp: lammps instance object.
    :param initial_conformation: lammps input data file with the particles initial conformation.
    :param CONFIG.neighbor neighbor: see LAMMPS_CONFIG.py.
    :param True hide_log: do not generate lammps log information
    :param FENE connectivity:   use {1     : ["FENE", 30.0, 1.5, 1.0, 1.0]} for a fene bond or harmonic for harmonic potential for neighbours
    :param LJ   excludedVolume: use {(1,1) : ["LJ", 1.0, 1.0, 1.12246152962189]}
    :param False restart_file: path to file to restore LAMMPs session (binary)

    """

    #if hide_log:
    #    lmp.command("log none")
    #os.remove("log.lammps")

    #######################################################
    # Box and units  (use LJ units and period boundaries) #
    #######################################################
    lmp.command("units %s" % CONFIG.units)
    lmp.command("atom_style %s" % CONFIG.atom_style) #with stiffness
    lmp.command("boundary %s" % CONFIG.boundary)
    #lmp.command("comm_style tiled")
    """
    try:
        lmp.command("communicate multi")
    except:
        pass
    """

    ##########################
    # READ "start" data file #
    ##########################
    if restart_file == False :
        lmp.command("read_data %s" % initial_conformation)
    else:
        restart_time = int(restart_file.split('/')[-1].split('_')[4][:-8])
        print('Previous unfinished LAMMPS steps found')
        print('Loaded %s file' %restart_file)
        lmp.command("read_restart %s" % restart_file)
        lmp.command("reset_timestep %i" % restart_time)
        
    lmp.command("mass %s" % CONFIG.mass)

    ##################################################################
    # Pair interactions require lists of neighbours to be calculated #
    ##################################################################
    lmp.command("neighbor %s" % neighbor)
    lmp.command("neigh_modify %s" % CONFIG.neigh_modify)
    
    ##############################################################
    # Sample thermodynamic info  (temperature, energy, pressure) #
    ##############################################################
    lmp.command("thermo %i" % CONFIG.thermo)
    #lmp.command("balance 1.1 rcb")
    
    ###############################
    # Stiffness term              #
    # E = K * (1+cos(theta)), K>0 #
    ###############################
    lmp.command("angle_style %s" % CONFIG.angle_style) # Write function for kinks     
    if persistence_length:
        if isinstance(persistence_length, (float)):
            lmp.command("angle_coeff * %f" % persistence_length)
        else:
            for i in range(len(persistence_length)):
                lmp.command("angle_coeff %i %f" % (i+1,persistence_length[i]))
    else:
        lmp.command("angle_coeff * %f" % CONFIG.persistence_length)
    
    ###################################################################
    # Pair interaction between non-bonded atoms                       #
    #                                                                 #
    #  Lennard-Jones 12-6 potential with cutoff:                      #
    #  potential E=4epsilon[ (sigma/r)^12 - (sigma/r)^6]  for r<r_cut #
    #  r_cut =1.12246 = 2^(1/6) is the minimum of the potential       #
    ###################################################################

    for atomTypePair in excludedVolume:

        lmp.command("pair_style hybrid/overlay lj/cut %f lj/cut %f" % (CONFIG.PurelyRepulsiveLJcutoff , CONFIG.PurelyRepulsiveLJcutoff))
        #lmp.command("pair_style lj/cut 1 %f" % CONFIG.PurelyRepulsiveLJcutoff)
        lmp.command("pair_coeff * * lj/cut 1 %f %f %f" % (excludedVolume[atomTypePair][1],
                                                          excludedVolume[atomTypePair][2],
                                                          excludedVolume[atomTypePair][3]))
        
        if excludedVolume[atomTypePair][1] == 0.0:
            #lmp.command("pair_style lj/cut 1 %f" % CONFIG.PurelyRepulsiveLJcutoff)
            lmp.command("pair_coeff * * lj/cut 1 %f %f %f" % (0.0, 0.0, 0.0))
            #lmp.command("pair_style lj/cut 2 %f" % CONFIG.PurelyRepulsiveLJcutoff)
            lmp.command("pair_coeff * * lj/cut 2 %f %f %f" % (0.0, 0.0, 0.0))                        
            continue
        
        #lmp.command("pair_style lj/cut 2 %f" % CONFIG.PurelyRepulsiveLJcutoff)
        lmp.command("pair_coeff * * lj/cut 2 %f %f %f" % (excludedVolume[atomTypePair][1],
                                                          excludedVolume[atomTypePair][2],
                                                          excludedVolume[atomTypePair][3]))
        
        ################################################################
        #  pair_modify shift yes adds a constant to the potential such #
        #  that E(r_cut)=0. Forces remains unchanged.                  #
        ################################################################
        lmp.command("pair_modify shift yes")
    
        ######################################
        #  pair_coeff for lj/cut, specify 4: #
        #    * atom type interacting with    #
        #    * atom type                     #
        #    * epsilon (energy units)        #
        #    * sigma (distance units)        #
        ######################################
        #lmp.command("pair_coeff %d %d %f %f %f" % (atomTypePair[0],
        #                                           atomTypePair[1],
        #                                           excludedVolume[atomTypePair][1],
        #                                           excludedVolume[atomTypePair][2],
        #                                           excludedVolume[atomTypePair][3]))
        

    for bondType in connectivity:
        if connectivity[bondType][0] == "FENE":
            #########################################################
            # Pair interaction between bonded atoms                 #
            #                                                       #
            # Fene potential + Lennard Jones 12-6:                  #
            #  E= - 0.5 K R0^2 ln[ 1- (r/R0)^2]                     #
            #     + 4epsilon[ (sigma/r)^12 - (sigma/r)^6] + epsilon #
            #########################################################
            lmp.command("bond_style fene")
            
            ########################################
            # For style fene, specify:             #
            #   * bond type                        #
            #   * K (energy/distance^2)            #
            #   * R0 (distance)                    #
            #   * epsilon (energy)  (LJ component) #
            #   * sigma (distance)  (LJ component) #
            ########################################
            lmp.command("bond_coeff %d %f %f %f %f" % (bondType, connectivity[bondType][1], connectivity[bondType][2], connectivity[bondType][3], connectivity[bondType][4]))
            lmp.command("special_bonds fene") #<=== I M P O R T A N T (new command)
            
        if connectivity[bondType][0] == "harmonic":
            lmp.command("bond_style harmonic")
            lmp.command("bond_coeff %d %f %f" % (bondType, connectivity[bondType][1], connectivity[bondType][2]))
        if connectivity[bondType][0] == "FENEspecial":
            lmp.command("bond_style fene")        
            lmp.command("bond_coeff %d %f %f %f %f" % (bondType, connectivity[bondType][1], connectivity[bondType][2], connectivity[bondType][3], connectivity[bondType][4]))
            lmp.command("special_bonds fene") #<=== I M P O R T A N T (new command)

    ##############################
    # set timestep of integrator #
    ##############################
    lmp.command("timestep %f" % timestep)

# This splits the lammps calculations on different processors:
def lammps_simulate(lammps_folder, run_time,
                    initial_conformation=None,
                    connectivity = {1:["FENE", 30.0, 1.5, 1.0, 1.0]},
                    excludedVolume = {(1,1):["LJ"  ,  1.0, 1.0, 1.12246152962189]},
                    nveLimit=False,
                    initial_seed=0, n_models=500, n_keep=100,
                    neighbor=CONFIG.neighbor, tethering=True,
                    add_external_force=None,
                    minimize=True, pushOff=False, compress_with_pbc=False,        
                    compress_without_pbc=False,
                    initial_relaxation=None,
                    restrained_dynamics=None,
                    keep_restart_out_dir=None, outfile=None, n_cpus=1,
                    confining_environment=['cube',300.],
                    steering_pairs=None,
                    time_dependent_steering_pairs=None,
                    compartmentalization=None,
                    loop_extrusion_dynamics_OLD=None, loop_extrusion_dynamics=None, cleanup = False,
                    to_dump=100000, pbc=False, timeout_job=3600,
                    hide_log=True,
                    gamma=CONFIG.gamma, timestep=CONFIG.timestep,
                    reset_timestep=0,
                    chromosome_particle_numbers=None,
                    restart_path=False,
                    store_n_steps=10,
                    useColvars=False):

    """
    This function launches jobs to generate three-dimensional models in lammps
    
    :param initial_conformation: structural _models object with the particles initial conformation. 
            http://lammps.sandia.gov/doc/2001/data_format.html
    :param FENE connectivity: use FENE for a fene bond or harmonic for harmonic potential
        for neighbours (see init_lammps for details)
    :param run_time: # of timesteps.
    :param None steering_pairs: dictionary with all the info to perform
            steered molecular dynamics.
            steering_pairs = { 'colvar_input'              : "ENST00000540866.2chr7_clean_enMatch.txt",
                               'colvar_output'             : "colvar_list.txt",
                               'kappa_vs_genomic_distance' : "kappa_vs_genomic_distance.txt",
                               'chrlength'                 : 3182,
                               'copies'                    : ['A'],
                               'binsize'                   : 50000,
                               'number_of_kincrease'       : 1000,
                               'timesteps_per_k'           : 1000,
                               'timesteps_relaxation'      : 100000,
                               'perc_enfor_contacts'       : 10
                             }
            Should at least contain Chromosome, loci1, loci2 as 1st, 2nd and 3rd column

    :param None loop_extrusion_dynamics: dictionary with all the info to perform loop
            extrusion dynamics.
            loop_extrusion_dynamics = { 'target_loops_input'          : "target_loops.txt",
                                        'loop_extrusion_steps_output' : "loop_extrusion_steps.txt",
                                        'attraction_strength'         : 4.0,
                                        'equilibrium_distance'        : 1.0,
                                        'chrlength'                   : 3182,
                                        'copies'                      : ['A'],
                                        'timesteps_per_loop_extrusion_step' : 1000,
                                        'timesteps_relaxation'        : 100000,
                                        'perc_enfor_loops'            : 10
                             }

            Should at least contain Chromosome, loci1, loci2 as 1st, 2nd and 3rd column 

    :param 0 initial_seed: Initial random seed for modelling
    :param 500 n_models: number of models to generate.
    :param CONFIG.neighbor neighbor: see LAMMPS_CONFIG.py.
    :param True minimize: whether to apply minimize command or not. 
    :param None keep_restart_out_dir: path to write files to restore LAMMPS session (binary)
    :param None outfile: store result in outfile
    :param 1 n_cpus: number of CPUs to use.
    :param False restart_path: path to files to restore LAMMPs session (binary)
    :param 10 store_n_steps: Integer with number of steps to be saved if 
        restart_file != False
    :param False useColvars: True if you want the restrains to be loaded by colvars

    :returns: a Tadphys models dictionary

    """
    
    if confining_environment[0] != 'cube' and pbc == True:
        print("ERROR: It is not possible to implement the pbc")
        print("for simulations inside a %s" % (confining_environment[0]))    

    if initial_seed:
        seed(initial_seed)

    #pool = mu.Pool(n_cpus)
    timepoints = 1
    if time_dependent_steering_pairs:
        timepoints = (len(time_dependent_steering_pairs['colvar_input'])-1) * \
            time_dependent_steering_pairs['colvar_dump_freq']

    #chromosome_particle_numbers = [int(x) for x in [len(LOCI)]]
    chromosome_particle_numbers.sort(key=int,reverse=True)

    kseeds = []
    for k in range(n_models):
        kseeds.append(k+1+initial_seed)
    #while len(kseeds) < n_models:
    #    rnd = randint(1,100000000)
    #    if all([(abs(ks - rnd) > timepoints) for ks in kseeds]):
    #        kseeds.append(rnd)

    #pool = ProcessPool(max_workers=n_cpus, max_tasks=n_cpus)
    pool = multiprocessing.Pool(processes=n_cpus, maxtasksperchild=n_cpus)

    results = []
    def collect_result(result):
        results.append((result[0], result[1], result[2]))

    initial_models = initial_conformation
    if not initial_models:
        initial_models = []

    jobs = {}
    for k_id, k in enumerate(kseeds):
        k_folder = lammps_folder + 'lammps_' + str(k) + '/'
        failedSeedLog = None
        # First we check if the modelling fails with this seed
        if restart_path != False:
            restart_file = restart_path + 'lammps_' + str(k) + '/'
            failedSeedLog = restart_file + 'runLog.txt'
            if os.path.exists(failedSeedLog):
                with open(failedSeedLog, 'r') as f:
                    for line in f:
                        prevRun = line.split()
                # add number of models done so dont repeat same seed
                if prevRun[1] == 'Failed':
                    k = int(prevRun[0]) + n_models
                    k_folder = lammps_folder + 'lammps_' + str(k) + '/'

        #print "#RandomSeed: %s" % k
        keep_restart_out_dir2 = None
        if keep_restart_out_dir != None:
            keep_restart_out_dir2 = keep_restart_out_dir + 'lammps_' + str(k) + '/'
            if not os.path.exists(keep_restart_out_dir2):
                os.makedirs(keep_restart_out_dir2)
        model_path = False
        if restart_path != False:
            # check presence of previously finished jobs
            model_path = restart_path + 'lammps_' + str(k) + '/finishedModel_%s.pickle' %k
        # define restart file by checking for finished jobs or last step
        if model_path != False and os.path.exists(model_path):
            with open(model_path, "rb") as input_file:
                m = load(input_file)
            results.append((m[0], m[1]))
        else:
            if restart_path != False:
                restart_file = restart_path + 'lammps_' + str(k) + '/'
                dirfiles = os.listdir(restart_file)
                # check for last k and step
                maxi = (0, 0, '')
                for f in dirfiles:
                    if f.startswith('restart_kincrease_'):
                        kincrease = int(f.split('_')[2])
                        step = int(f.split('_')[-1][:-8])
                        if kincrease > maxi[0]:
                            maxi = (kincrease, step, f)
                        elif kincrease == maxi[0] and step > maxi[1]:
                            maxi = (kincrease, step, f)
                # In case there is no restart file at all
                if maxi[2] == '':
                    #print('Could not find a LAMMPS restart file')
                    # will check later if we have a path or a file
                    getIniConf = True
                    #restart_file = False
                else:
                    restart_file = restart_file + maxi[2]
                    getIniConf = False
            else:
                restart_file = False
                getIniConf = True

            ini_conf = None
            if not os.path.exists(k_folder):
                os.makedirs(k_folder)
                if initial_conformation and getIniConf == True:
                    ini_conf = '%sinitial_conformation.dat' % k_folder
                    write_initial_conformation_file(initial_conformation[k_id],
                                                    chromosome_particle_numbers,
                                                    confining_environment,
                                                    out_file=ini_conf)
    #         jobs[k] = run_lammps(k, k_folder, run_time,
    #                                               initial_conformation, connectivity,
    #                                               neighbor,
    #                                               tethering, minimize, pushOff,
    #                                               compress_with_pbc, compress_without_pbc,
    #                                               confining_environment,
    #                                               steering_pairs,
    #                                               time_dependent_steering_pairs,
    #                                               loop_extrusion_dynamics,
    #                                               to_dump, pbc,)
    #       jobs[k] = pool.schedule(run_lammps,
            jobs[k] = partial(abortable_worker, run_lammps, timeout=timeout_job,
                                failedSeedLog=[failedSeedLog, k])
            pool.apply_async(jobs[k],
                            args=(k, k_folder, run_time,
                                  ini_conf, connectivity,
                                  neighbor,
                                  tethering, add_external_force, minimize, pushOff,                           
                                  compress_with_pbc, compress_without_pbc,
                                  initial_relaxation,
                                  restrained_dynamics,
                                  confining_environment,
                                  steering_pairs,
                                  time_dependent_steering_pairs,
                                  compartmentalization,
                                  loop_extrusion_dynamics,
                                  to_dump, pbc, hide_log,                                  
                                  gamma,timestep,reset_timestep,
                                  chromosome_particle_numbers,
                                  keep_restart_out_dir2,
                                  restart_file,
                                  model_path,
                                  store_n_steps,
                                  useColvars,), callback=collect_result)
            #                         , timeout=timeout_job)
    pool.close()
    pool.join()

#     for k in kseeds:
#         try:
#             #results.append((k, jobs[k]))
#             results.append((k, jobs[k].result()))
#         except TimeoutError:
#             print "Model took more than %s seconds to complete ... canceling" % str(timeout_job)
#             jobs[k].cancel()
#         except Exception as error:
#             print "Function raised %s" % error
#             jobs[k].cancel()

    models = {}
    initial_models = []
    ############ WARNING ############
    # PENDING TO ADD THE STORAGE OF INITIAL MODELS #
    if timepoints > 1:
        for t in range(timepoints):
            time_models = []
            for res in results:
                (k,restarr,init_conf) = res
                time_models.append(restarr[t])
            for i, m in enumerate(time_models[:n_keep]):
                models[i+t*len(time_models[:n_keep])+n_keep] = m
            #for i, (_, m) in enumerate(
            #    sorted(time_models.items(), key=lambda x: x[1]['objfun'])[:n_keep]):
            #    models[i+t+1] = m

    else:
        for i, (_, m, im) in enumerate(
            sorted(results, key=lambda x: x[1][0]['objfun'])[:n_keep]):
            models[i] = m[0]
            if not initial_conformation:
                initial_models += [im]

    if cleanup:
        for k in kseeds:
            k_folder = lammps_folder + '/lammps_' + str(k) + '/'
            if os.path.exists(k_folder):
                shutil.rmtree(k_folder)

    return models, initial_models
    
    
# This performs the dynamics: I should add here: The steered dynamics (Irene and Hi-C based) ; 
# the binders based dynamics (Marenduzzo and Nicodemi)...etc...
def run_lammps(kseed, lammps_folder, run_time,
               initial_conformation=None,
               connectivity={1:["FENE", 30.0, 1.5, 1.0, 1.0]},
               excludedVolume = {(1,1):["LJ"  ,  1.0, 1.0, 1.12246152962189]},
               nveLimit=False,
               persistence_length=0.00,
               neighbor=CONFIG.neighbor,
               fixed_particles=None,
               tethering=False, add_external_force=None, minimize=True, pushOff=False,
               compress_with_pbc=None, compress_without_pbc=None,
               initial_relaxation=None,
               restrained_dynamics=None,
               confining_environment=None,
               steering_pairs=None,
               time_dependent_steering_pairs=None,
               compartmentalization=None,
               loop_extrusion_dynamics_OLD=None,
               loop_extrusion_dynamics=None,               
               transcription_based_loop_extrusion_dynamics=None,               
               to_dump=10000, pbc=False,
               hide_log=True,
               gamma=CONFIG.gamma, timestep=CONFIG.timestep,
               reset_timestep=0,
               chromosome_particle_numbers=None,
               keep_restart_out_dir2=None,
               restart_file=False,
               model_path=False, 
               store_n_steps=10,
               useColvars=False):
    """
    Generates one lammps model
    
    :param kseed: Random number to identify the model.
    :param initial_conformation_folder: folder where to store lammps input 
        data file with the particles initial conformation. 
        http://lammps.sandia.gov/doc/2001/data_format.html
    :param FENE connectivity: use FENE for a fene bond or harmonic for harmonic
        potential for neighbours (see init_lammps_run) 
    :param run_time: # of timesteps.
    :param None initial_conformation: path to initial conformation file or None 
        for random walk initial start.
    :param CONFIG.neighbor neighbor: see LAMMPS_CONFIG.py.
    :param False tethering: whether to apply tethering command or not.
    :param True minimize: whether to apply minimize command or not. 
    :param None compress_with_pbc: whether to apply the compression dynamics in case of a
      system with cubic confinement and pbc. This compression step is usually apply 
      to obtain a system with the desired particle density. The input have to be a list 
      of three elements:
      0 - XXX;
      1 - XXX;
      2 - The compression simulation time span (in timesteps).
      e.g. compress_with_pbc=[0.01, 0.01, 100000]
    :param None compress_without_pbc: whether to apply the compression dynamics in case of a
      system with spherical confinement. This compression step is usually apply to obtain a 
      system with the desired particle density. The simulation shrinks/expands the initial 
      sphere to a sphere of the desired radius using many short runs. In each short run the
      radius is reduced by 0.1 box units. The input have to be a list of three elements:
      0 - Initial radius;
      1 - Final desired radius;
      2 - The time span (in timesteps) of each short compression run.
      e.g. compress_without_pbc=[300, 100, 100]
    :param None steering_pairs: particles contacts file from colvars fix 
      http://lammps.sandia.gov/doc/PDF/colvars-refman-lammps.pdf. 
      steering_pairs = { 'colvar_input'              : "ENST00000540866.2chr7_clean_enMatch.txt",
                         'colvar_output'             : "colvar_list.txt",
                         'kappa_vs_genomic_distance' : "kappa_vs_genomic_distance.txt",
                         'chrlength'                 : 3182,
                         'copies'                    : ['A'],
                         'binsize'                   : 50000,
                         'number_of_kincrease'       : 1000,
                         'timesteps_per_k'           : 1000,
                         'timesteps_relaxation'      : 100000,
                         'perc_enfor_contacts'       : 10
                       }

    :param None loop_extrusion_dynamics: dictionary with all the info to perform loop 
            extrusion dynamics.
            loop_extrusion_dynamics = { 'target_loops_input'          : "target_loops.txt",
                                        'loop_extrusion_steps_output' : "loop_extrusion_steps.txt",
                                        'attraction_strength'         : 4.0,
                                        'equilibrium_distance'        : 1.0,
                                        'chrlength'                   : 3182,
                                        'copies'                      : ['A'],
                                        'timesteps_per_loop_extrusion_step' : 1000,
                                        'timesteps_relaxation'        : 100000,
                                        'perc_enfor_loops'            : 10
                             }

            Should at least contain Chromosome, loci1, loci2 as 1st, 2nd and 3rd column 
    :param None keep_restart_out_dir2: path to write files to restore LAMMPs
                session (binary)
    :param False restart_file: path to file to restore LAMMPs session (binary)
    :param False model_path: path to/for pickle with finished model (name included)
    :param 10 store_n_steps: Integer with number of steps to be saved if 
        restart_file != False
    :param False useColvars: True if you want the restrains to be loaded by colvars
    :returns: a LAMMPSModel object

    """

    lmp = lammps(cmdargs=['-screen','none','-log',lammps_folder+'log.lammps','-nocite'])
    me = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()
    # check if we have a restart file or a path to which restart
    if restart_file == False:
        doRestart = False
        saveRestart = False
    elif os.path.isdir(restart_file):
        doRestart = False
        saveRestart = True
    else:
        doRestart = True
        saveRestart = True
    if not initial_conformation and doRestart == False:    
        initial_conformation = lammps_folder+'initial_conformation.dat'
        generate_chromosome_random_walks_conformation ([len(LOCI)],
                                                       outfile=initial_conformation,
                                                       seed_of_the_random_number_generator=randint(1,100000),
                                                       confining_environment=confining_environment)
    
    # Just prepared the steps recovery for steering pairs
    if steering_pairs and doRestart == True:
        init_lammps_run(lmp, initial_conformation,
                        neighbor=neighbor,
                        hide_log=hide_log,
                        timestep=timestep,
                        reset_timestep=reset_timestep,
                        connectivity=connectivity,
                        excludedVolume=excludedVolume,
                        nveLimit=nveLimit,
                        persistence_length=persistence_length,
                        restart_file=restart_file)
    else:
        init_lammps_run(lmp, initial_conformation,
                        neighbor=neighbor,
                        hide_log=hide_log,
                        timestep=timestep,
                        reset_timestep=reset_timestep,
                        chromosome_particle_numbers=chromosome_particle_numbers,
                        connectivity=connectivity,
                        excludedVolume=excludedVolume,
                        nveLimit=nveLimit,
                        persistence_length=persistence_length)
        
    lmp.command("dump    1       all    custom    %i   %slangevin_dynamics_*.XYZ  id  xu yu zu" % (to_dump,lammps_folder))
    lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id")


    # ##########################################################
    # # Generate RESTART file, SPECIAL format, not a .txt file #
    # # Useful if simulation crashes             
    # Prepared an optimisation for steering pairs, but not for the rest#
    # ##########################################################
    # create lammps restart files every x steps. 1000 is ok
    # There was the doubt of using text format session info (which allows use in other computers)
    # but since the binary can be converted later and this: "Because a data file is in text format, 
    # if you use a data file written out by this command to restart a simulation, the initial state 
    # of the new run will be slightly different than the final state of the old run (when the file 
    # was written) which was represented internally by LAMMPS in binary format. A new simulation 
    # which reads the data file will thus typically diverge from a simulation that continued 
    # in the original input script." will continue with binary. To convert use restart2data
    #if keep_restart_out_dir2:
    #    lmp.command("restart %i %s/relaxation_%i_*.restart" % (keep_restart_step, keep_restart_out_dir2, kseed))


    #######################################################
    # Set up fixes                                        #
    # use NVE ensemble                                    #
    # Langevin integrator Tstart Tstop 1/friction rndseed #
    # => sampling NVT ensamble                            #
    #######################################################
    # Set the group of particles that will be moved during the molecular dynamics
    nparticles = int(lmp.get_natoms())
    print("Number of atoms in the system =",nparticles)

    #fixed_group  = ""
    #mobile_group = ""
    #if not fixed_particles:
    #    fixed_particles = []
    #for particle in range(1,nparticles+1):
    #    if particle in fixed_particles:
    #        fixed_group += "%d " % (particle)
    #        print("set atom %d vx 0.0 vy 0.0 vz 0.0" % particle)
    #        lmp.command("set atom %d vx 0.0 vy 0.0 vz 0.0" % particle)
    #    else:
    #        mobile_group += "%d " % (particle)
            
    #print("group mobile_particles id %s" % mobile_group)
    #lmp.command("group mobile_particles id %s" % mobile_group)

    #if fixed_particles != []:
    #    print("group fixed_particles id %s" % fixed_group)
    #    lmp.command("group fixed_particles id %s" % fixed_group)
    #    print("fix freeze fixed_particles setforce 0.0 0.0 0.0")
    #    lmp.command("fix freeze fixed_particles setforce 0.0 0.0 0.0")

    # Define the langevin dynamics integrator
    #lmp.command("fix 1 mobile_particles nve")
    #if type(nveLimit) is list:
    #    lmp.command("unfix 1")
    #    lmp.command("fix 1 mobile_particles nve/limit %f" % nveLimit[0])
    #lmp.command("fix 2 mobile_particles langevin 1.0  1.0  %f %i" % (gamma,randint(1,100000)))
    lmp.command("fix 1 all nve")
    if type(nveLimit) is list:
        lmp.command("unfix 1")
        lmp.command("fix 1 all nve/limit %f" % nveLimit[0])
    lmp.command("fix 2 all langevin 1.0  1.0  %f %i" % (gamma,randint(1,100000)))    

    # Define the tethering to the center of the confining environment
    if tethering:
        lmp.command("fix 3 all spring tether 50.0 0.0 0.0 0.0 0.0")

    if confining_environment:
        if confining_environment[0] == "sphere":

            radius  = confining_environment[1]
            if(len(confining_environment) < 3):
                xcentre = 0.0
                ycentre = 0.0
                zcentre = 0.0
                eps     = 1.0
                sig     = 1.0
            else:
                xcentre = confining_environment[2]
                ycentre = confining_environment[3]
                zcentre = confining_environment[4]
                eps     = confining_environment[5]
                sig     = confining_environment[6]
                
            lmp.command("region sphere sphere %f %f %f %f units box side in" % (xcentre, ycentre, zcentre, radius))
            print("region sphere sphere %f %f %f %f units box side in" % (xcentre, ycentre, zcentre, radius))
            
            # Performing the simulation
            
            lmp.command("fix 5 all  wall/region sphere lj126 %f %f %f" % (eps, sig, sig*1.12246152962189))
            print("fix 5 all  wall/region sphere lj126 %f %f %f" % (eps, sig, sig*1.12246152962189))
            
    if add_external_force:
        # add_external_force is a dictionary with elements particle : [fx,fy,fz]
        nForces = 1
        for particle in add_external_force:
            lmp.command("group extForce%d id %d" % (nForces, particle))            
            lmp.command("fix extForce%d extForce%d addforce %f %f %f" % (nForces, nForces, add_external_force[particle][0],add_external_force[particle][1],add_external_force[particle][2]))
            nForces = nForces + 1
        
    # Do a minimization step to prevent particles
    # clashes in the initial conformation
    if minimize:

        if to_dump:
            lmp.command("undump 1")
            lmp.command("dump    1       all    custom    %i   %sminimization_*.XYZ  id  xu yu zu" % (to_dump,lammps_folder))
            #lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id append yes")
        
        print("Performing minimization run...")
        lmp.command("minimize 1.0e-6 1.0e-8 10000000 10000000")
        #lmp.command("minimize 1.0e-4 1.0e-6 100000 100000")
        
        if to_dump:
            lmp.command("undump 1")
            lmp.command("dump    1       all    custom    %i   %slangevin_dynamics_*.XYZ  id  xu yu zu" % (to_dump,lammps_folder))
            lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id")        
        lmp.command("reset_timestep 0") 
            
    if reset_timestep != 0:
        lmp.command("reset_timestep %i" % (reset_timestep))         

    if pushOff:
        # A_initial (energy units) -> pushOff[0]
        # A_final   (energy units) -> pushOff[1]
        # cutoff (distance units)  -> pushOff[2]
        # E.g. pushOff = [0,100,1.0]
        
        lmp.command("pair_style soft %f" % (pushOff[2]))
        lmp.command("pair_coeff * * 0.0")
        lmp.command("variable prefactor equal ramp(%f,%f)" % (pushOff[0],pushOff[1]))
        lmp.command("fix pushOff all adapt 1 pair soft a * * v_prefactor")

        # epsilon (energy units) -> pushOff[0]
        # sigma   (energy units) -> pushOff[1]
        # lambda (activation parameter, between 1 and 1) -> pushOff[2]
        # cutoff  (distance units) -> pushOff[3]
        # E.g. pushOff = [1.0,1.0,1.0,1.12246152962189]
        
        #lmp.command("pair_style lj/cut/soft %f" % (pushOff[3]))
        #print("pair_coeff * * %f %f %f %f" % (pushOff[0],pushOff[1],pushOff[2],pushOff[3]))        
        #lmp.command("pair_coeff * * %f %f %f %f" % (pushOff[0],pushOff[1],pushOff[2],pushOff[3]))        
        #lmp.command("pair_modify shift yes")
        
    if compress_with_pbc:
        lmp.command("velocity all create 1.0 %s" % randint(1,100000))
        try:
            fixed_extruders = transcription_based_loop_extrusion_dynamics['fixed_extruders']
            print("Define the fixed extruders once for all")
            fixed_extruder_number=0
            for particle1,particle2 in fixed_extruders:
                fixed_extruder_number += 1
                print("# fix fixed_LE%i all restrain bond %i  %i %f %f %f" % (fixed_extruder_number,
                                                                              particle1,
                                                                              particle2,
                                                                              transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
                                                                              transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
                                                                              transcription_based_loop_extrusion_dynamics['equilibrium_distance']))
            
                lmp.command("fix fixed_LE%i all restrain bond %i  %i %f %f %f" % (fixed_extruder_number,
                                                                                  particle1,
                                                                                  particle2,
                                                                                  transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
                                                                                  transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
                                                                                  transcription_based_loop_extrusion_dynamics['equilibrium_distance']))
                
            print("Defined",fixed_extruder_number,"fixed extruders")
        except:
            pass
        
        if to_dump:
            lmp.command("undump 1")
            lmp.command("dump    1       all    custom    %i   %scompress_with_pbc_*.XYZ  id  xu yu zu" % (to_dump,lammps_folder))
            lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id append no")

        # Re-setting the initial timestep to 0
        lmp.command("reset_timestep 0")

        lmp.command("unfix 1")
        lmp.command("unfix 2")

        lmp.command("thermo %d" % (compress_with_pbc[2]/100))
        lmp.command("thermo_style   custom   step temp etotal pxx pyy pzz pxy pxz pyz xlo xhi ylo yhi zlo zhi")
        
        # default as in PLoS Comp Biol Di Stefano et al. 2013 compress_with_pbc = [0.01, 0.01, 100000]
        lmp.command("fix 1 all   nph   iso   %s %s   2.0" % (compress_with_pbc[0], 
                                                             compress_with_pbc[1]))
        #lmp.command("fix 2 mobile_particles langevin 1.0  1.0  %f %i" % (gamma,randint(1,100000)))
        lmp.command("fix 2 all langevin 1.0  1.0  %f %i" % (gamma,randint(1,100000)))        
        print("run %i" % compress_with_pbc[2])
        lmp.command("run %i" % compress_with_pbc[2])

        lmp.command("unfix 1")
        lmp.command("unfix 2")

        #lmp.command("fix 1 mobile_particles nve")
        #lmp.command("fix 2 mobile_particles langevin 1.0  1.0  %f %i" % (gamma,randint(1,100000)))
        lmp.command("fix 1 all nve")
        lmp.command("fix 2 all langevin 1.0  1.0  %f %i" % (gamma,randint(1,100000)))        

        lmp.command("write_data compressed_conformation.txt nocoeff")
        
        # Here We have to re-define the confining environment
        print("# Previous particle density (nparticles/volume) %f" % (int(lmp.get_natoms())/(float(confining_environment[1])**3)))
        confining_environment[1] = lmp.extract_box()[1][0] - lmp.extract_box()[0][0]
        print("")
        print("# New cubic box dimensions after isotropic compression")
        print(lmp.extract_box()[0][0],lmp.extract_box()[1][0])
        print(lmp.extract_box()[0][1],lmp.extract_box()[1][1])
        print(lmp.extract_box()[0][2],lmp.extract_box()[1][2])
        print("# New confining environment", confining_environment)
        print("# New volumetric density (Vpolymer/Vbox) %f" % (int(lmp.get_natoms())*0.5*0.5*0.5*4./3.*pi/(float(confining_environment[1])**3)))
        print("")

        if to_dump:
            lmp.command("undump 1")
            lmp.command("dump    1       all    custom    %i   %slangevin_dynamics_*.XYZ  id  xu yu zu" % (to_dump,lammps_folder))
            lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id")        

            
    if compress_without_pbc:
        print("Performing compression within a sphere without pbc")
        
        #fixed_extruders = transcription_based_loop_extrusion_dynamics['fixed_extruders']
        #print("Define the fixed extruders once for all")
        #fixed_extruder_number=0
        #for particle1,particle2 in fixed_extruders:
        #    fixed_extruder_number += 1
        #    print("# fix fixed_LE%i all restrain bond %i  %i %f %f %f" % (fixed_extruder_number,
        #                                                                  particle1,
        #                                                                  particle2,
        #                                                                  transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
        #                                                                  transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
        #                                                                  transcription_based_loop_extrusion_dynamics['equilibrium_distance']))
            
        #    lmp.command("fix fixed_LE%i all restrain bond %i  %i %f %f %f" % (fixed_extruder_number,
        #                                                                      particle1,
        #                                                                      particle2,
        #                                                                      transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
        #                                                                      transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
        #                                                                      transcription_based_loop_extrusion_dynamics['equilibrium_distance']))

            
        #print("Defined",fixed_extruder_number,"fixed extruders")
        
        if to_dump:
            lmp.command("undump 1")
            lmp.command("dump    1       all    custom    %i   %scompress_without_pbc_*.XYZ  id  xu yu zu " % (int(compress_without_pbc[5]/10),lammps_folder))
            lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id")

        # Re-setting the initial timestep to 0
        lmp.command("reset_timestep 0")

        eps     = confining_environment[5]
        sig     = confining_environment[6]
        
        # default as in Sci Rep Di Stefano et al. 2016 
        # compress_without_pbc = [xcentre, ycentre, zcentre, initial_radius, final_radius, runtime] 
        # = [350, 161.74, 100]
        
        print("variable NuclearRadius equal %f-step/%f*%f" % (compress_without_pbc[3],compress_without_pbc[5],compress_without_pbc[4]))
        print("region nucleus sphere %f %f %f v_NuclearRadius units box side in" % (compress_without_pbc[0],compress_without_pbc[1],compress_without_pbc[2]))
        lmp.command("variable NuclearRadius equal %f-step/%f*(%f-%f)" % (compress_without_pbc[3],compress_without_pbc[5],compress_without_pbc[3],compress_without_pbc[4]))
        lmp.command("region nucleus sphere %f %f %f v_NuclearRadius units box side in" % (compress_without_pbc[0],compress_without_pbc[1],compress_without_pbc[2]))

        lmp.command("unfix 5")
        lmp.command("fix 5 all  wall/region nucleus lj126 %f %f %f" % (eps, sig, sig*1.12246152962189))
        print("fix 5 all  wall/region nucleus lj126 %f %f %f" % (eps, sig, sig*1.12246152962189))
        
        lmp.command("thermo %d" % int(compress_without_pbc[5]/100))
        lmp.command("thermo_style   custom   step temp etotal pxx pyy pzz pxy pxz pyz v_NuclearRadius")
        
        print("run %d" % compress_without_pbc[5])
        lmp.command("run %d" % compress_without_pbc[5])
        
        # Here we have to re-define the confining environment
        volume = 4.*np.pi/3.0*float((compress_without_pbc[3]**3))
        print("# Previous particle density (nparticles/volume) %f" % (int(lmp.get_natoms())/volume))
        confining_environment[1] = compress_without_pbc[4]
        print("")
        volume = 4.*np.pi/3.0*float((compress_without_pbc[4]**3))
        print("# New particle density (nparticles/volume) %f" % (int(lmp.get_natoms())/volume))
        print("")
        print("region nucleus delete")
        lmp.command("region nucleus delete")        
        print("region nucleus sphere %f %f %f %f units box side in" % (compress_without_pbc[0],compress_without_pbc[1],compress_without_pbc[2],compress_without_pbc[4]))        
        lmp.command("region nucleus sphere %f %f %f %f units box side in" % (compress_without_pbc[0],compress_without_pbc[1],compress_without_pbc[2],compress_without_pbc[4]))

        lmp.command("unfix 5")
        lmp.command("fix 5 all  wall/region nucleus lj126 %f %f %f" % (eps, sig, sig*1.12246152962189))
        print("fix 5 all  wall/region nucleus lj126 %f %f %f" % (eps, sig, sig*1.12246152962189))        
        
        if to_dump:
            lmp.command("undump 1")
            lmp.command("dump    1       all    custom    %i   %slangevin_dynamics_*.XYZ  id  xu yu zu" % (to_dump,lammps_folder))
            lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id")
        
    if initial_relaxation:
        
        if to_dump:
            lmp.command("undump 1")
            #lmp.command("dump    1       all    custom    %i   %sinitial_relaxation.XYZ  id type  xu yu zu" % (to_dump,lammps_folder))
            #lmp.command("dump_modify     1 format line \"%d %d %.5f %.5f %.5f\" sort id append yes")
            lmp.command("dump    1       all    custom    %i   %sinitial_relaxation.XYZ  id xu yu zu" % (to_dump,lammps_folder))
            lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id append yes")
        if "MSD" in initial_relaxation:
            lmp.command("compute MSD all msd")
            lmp.command("variable MSD equal c_MSD[4]")
            #lmp.command("variable dx  equal c_MSD[1]")
            #lmp.command("variable dy  equal c_MSD[2]")
            #lmp.command("variable dz  equal c_MSD[3]")
            lmp.command("variable step  equal step")
            #lmp.command("fix MSD all print %i \"${step} ${dx} ${dy} ${dz} ${MSD}\" file MSD.txt" % (initial_relaxation["MSD"]))
            lmp.command("fix MSD all print %i \"${step} ${MSD}\" file MSD.txt" % (initial_relaxation["MSD"]))

        if 'distances' in initial_relaxation:
            #lmp.command("compute xu all property/atom xu")
            #lmp.command("compute yu all property/atom yu")
            #lmp.command("compute zu all property/atom zu")
            #pair_number = 0
            #for particle1 in range(1,chrlength[0]):
            #    for particle2 in range(particle1,chrlength[0]):
            #        pair_number += 1
            #        lmp.command("variable x%i equal c_xu[%i]" % (particle1, particle1))
            #        lmp.command("variable x%i equal c_xu[%i]" % (particle2, particle2))
            #        lmp.command("variable y%i equal c_yu[%i]" % (particle1, particle1))
            #        lmp.command("variable y%i equal c_yu[%i]" % (particle2, particle2))
            #        lmp.command("variable z%i equal c_zu[%i]" % (particle1, particle1))
            #        lmp.command("variable z%i equal c_zu[%i]" % (particle2, particle2))
                    
            #        lmp.command("variable xLE%i equal v_x%i-v_x%i" % (pair_number, particle1, particle2))
            #        lmp.command("variable yLE%i equal v_y%i-v_y%i" % (pair_number, particle1, particle2))
            #        lmp.command("variable zLE%i equal v_z%i-v_z%i" % (pair_number, particle1, particle2))
            #        lmp.command("variable dist_%i_%i equal sqrt(v_xLE%i*v_xLE%i+v_yLE%i*v_yLE%i+v_zLE%i*v_zLE%i)" % (particle1, particle2, pair_number, pair_number, pair_number, pair_number, pair_number, pair_number))

            lmp.command("compute pairs     all property/local patom1 patom2")
            lmp.command("compute distances all pair/local dist")
            #lmp.command("variable distances equal c_distances[1]")
            lmp.command("dump distances all local %i distances.txt c_pairs[1] c_pairs[2] c_distances" % (initial_relaxation['distances']))
            #lmp.command("fix distances all print %i \"${step} ${distances}\" file distances.txt" % (initial_relaxation['distances']))
            
        lmp.command("reset_timestep 0")
        lmp.command("run %i" % initial_relaxation["relaxation_time"])
        lmp.command("write_data relaxed_conformation.txt nocoeff")
        if "MSD" in initial_relaxation:
            lmp.command("uncompute MSD")
        if "distances" in initial_relaxation:
            lmp.command("uncompute distances")
            lmp.command("uncompute pairs")
            lmp.command("undump    distances")

    if restrained_dynamics:
        #restrained_dynamics = {
        #    'harmonic_restraints_file'            : 'harmonic_restraints.txt'
        #    'lowerBound_harmonic_restraints_file' : 'lowerBound_harmonic_restraints.txt'
        #}

        lmp.command("compute xu all property/atom xu")
        lmp.command("compute yu all property/atom yu")
        lmp.command("compute zu all property/atom zu")

        if "harmonic_restraints_file"in restrained_dynamics:
            #add_harmonic_restraints(restrained_dynamics["harmonic_restraints_file"])
            restraints = restrained_dynamics["harmonic_restraints_file"]

            # particle1 particle2 k0 kf d0 df
            fp = open(restraints, "r")
            
            thermo_style="thermo_style   custom   step temp epair emol"
            restrain_number = 0
            for line in fp.readlines():
            
                line = line.strip().split()
                if line[0] == "#":
                    continue
                if len(line) != 6:
                    print("ERROR in",line,"6 fields expected: particle1 particle2 k_init k_final d_init d_final")


                #bond args = atom1 atom2 Kstart Kstop r0start (r0stop)
                #atom1,atom2 = IDs of 2 atoms in bond
                #Kstart,Kstop = restraint coefficients at start/end of run (energy units)
                #r0start = equilibrium bond distance at start of run (distance units)
                #r0stop = equilibrium bond distance at end of run (optional) (distance units). If not
                #specified it is assumed to be equal to r0start
                particle1 = int(line[0])
                particle2 = int(line[1])
                lmp.command("fix RESTR%i all restrain bond %i %i %f %f %f %f" % (restrain_number,
                                                                                 particle1,
                                                                                 particle2,
                                                                                 abs(float(line[2])),
                                                                                 abs(float(line[3])),
                                                                                 float(line[4]),
                                                                                 float(line[5])))

                lmp.command("variable x%i equal c_xu[%i]" % (particle1, particle1))
                lmp.command("variable x%i equal c_xu[%i]" % (particle2, particle2))
                lmp.command("variable y%i equal c_yu[%i]" % (particle1, particle1))
                lmp.command("variable y%i equal c_yu[%i]" % (particle2, particle2))
                lmp.command("variable z%i equal c_zu[%i]" % (particle1, particle1))
                lmp.command("variable z%i equal c_zu[%i]" % (particle2, particle2))
                
                lmp.command("variable xH%i equal v_x%i-v_x%i" % (restrain_number,
                                                                       particle1,
                                                                       particle2))
                lmp.command("variable yH%i equal v_y%i-v_y%i" % (restrain_number,
                                                                       particle1,
                                                                       particle2))
                lmp.command("variable zH%i equal v_z%i-v_z%i" % (restrain_number,
                                                                       particle1,
                                                                       particle2))
                lmp.command("variable Hdist_%i_%i equal sqrt(v_xH%i*v_xH%i+v_yH%i*v_yH%i+v_zH%i*v_zH%i)" % (particle1,
                                                                                                            particle2,
                                                                                                            restrain_number,
                                                                                                            restrain_number,
                                                                                                            restrain_number,
                                                                                                            restrain_number,
                                                                                                            restrain_number,
                                                                                                            restrain_number))

                thermo_style += " v_Hdist_%i_%i" % (particle1,
                                                    particle2)

                restrain_number += 1
                
        if "lowerBound_harmonic_restraints_file" in restrained_dynamics:
            #add_lowerBound_harmonic_restraints(restrained_dynamics["lowerBound_harmonic_restraints_file"])
            restraints = restrained_dynamics["lowerBound_harmonic_restraints_file"]
            fp = open(restraints, "r")
            
            restrain_number = 0
            for line in fp.readlines():
            
                line = line.strip().split()
                if line[0] == "#":
                    continue
                if len(line) != 6:
                    print("ERROR in",line,"6 fields expected: particle1 particle2 k_init k_final d_init d_final")


                #lbound args = atom1 atom2 Kstart Kstop r0start (r0stop)
                #atom1,atom2 = IDs of 2 atoms in bond
                #Kstart,Kstop = restraint coefficients at start/end of run (energy units)
                #r0start = equilibrium bond distance at start of run (distance units)
                #r0stop = equilibrium bond distance at end of run (optional) (distance units). If not
                #specified it is assumed to be equal to r0start
                particle1 = int(line[0])
                particle2 = int(line[1])
                lmp.command("fix LW_RESTR%i all restrain lbound %i %i %f %f %f %f" % (restrain_number,
                                                                                      particle1,
                                                                                      particle2,
                                                                                      abs(float(line[2])),
                                                                                      abs(float(line[3])),
                                                                                      float(line[4]),
                                                                                      float(line[5])))

                lmp.command("variable x%i equal c_xu[%i]" % (particle1, particle1))
                lmp.command("variable x%i equal c_xu[%i]" % (particle2, particle2))
                lmp.command("variable y%i equal c_yu[%i]" % (particle1, particle1))
                lmp.command("variable y%i equal c_yu[%i]" % (particle2, particle2))
                lmp.command("variable z%i equal c_zu[%i]" % (particle1, particle1))
                lmp.command("variable z%i equal c_zu[%i]" % (particle2, particle2))
                
                lmp.command("variable xLB%i equal v_x%i-v_x%i" % (restrain_number,
                                                                       particle1,
                                                                       particle2))
                lmp.command("variable yLB%i equal v_y%i-v_y%i" % (restrain_number,
                                                                       particle1,
                                                                       particle2))
                lmp.command("variable zLB%i equal v_z%i-v_z%i" % (restrain_number,
                                                                       particle1,
                                                                       particle2))
                lmp.command("variable LBdist_%i_%i equal sqrt(v_xLB%i*v_xLB%i+v_yLB%i*v_yLB%i+v_zLB%i*v_zLB%i)" % (particle1,
                                                                                                            particle2,
                                                                                                            restrain_number,
                                                                                                            restrain_number,
                                                                                                            restrain_number,
                                                                                                            restrain_number,
                                                                                                            restrain_number,
                                                                                                            restrain_number))

                thermo_style += " v_LBdist_%i_%i" % (particle1,
                                                     particle2)

        print(thermo_style)
        lmp.command("%s" % thermo_style)
        lmp.command("run %i" % run_time)

    if transcription_based_loop_extrusion_dynamics:

        # Start relaxation dynamics
        if transcription_based_loop_extrusion_dynamics['timesteps_relaxation'] > 0:

            lmp.command("reset_timestep 0")
            initial_reset_timestep = 0

            if initial_conformation[0:18] == "restart_relaxation":
                initial_reset_timestep = int(initial_conformation.split("_")[2].split(".")[0])
                reset_timestep = initial_reset_timestep
                print(initial_reset_timestep)
            elif initial_conformation[0:22] == "restart_loop_extrusion":
                initial_reset_timestep = transcription_based_loop_extrusion_dynamics['timesteps_relaxation']

            if transcription_based_loop_extrusion_dynamics['timesteps_relaxation'] - initial_reset_timestep > 0:
                lmp.command("undump 1")
                lmp.command("dump    1       all    custom    %i   relaxation_MD_*.XYZ  id  xu yu zu" % transcription_based_loop_extrusion_dynamics['to_dump_relaxation'])
                lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id append yes")

                for reset_timestep in range(initial_reset_timestep, transcription_based_loop_extrusion_dynamics['timesteps_relaxation']-transcription_based_loop_extrusion_dynamics['keep_restart_step']+1,transcription_based_loop_extrusion_dynamics['keep_restart_step']):

                    print("reset_timestep %d" % initial_reset_timestep)
                    #print "write_restart restart_relaxation_%i.restart" % (transcription_based_loop_extrusion_dynamics['keep_restart_step'])
                    print("run %i" % transcription_based_loop_extrusion_dynamics['keep_restart_step'])
                    lmp.command("reset_timestep %d" % reset_timestep)
                    lmp.command("run %i" % transcription_based_loop_extrusion_dynamics['keep_restart_step'])
                    #lmp.command("write_restart restart_relaxation_%i.restart" % (transcription_based_loop_extrusion_dynamics['keep_restart_step']))

                    lmp.command("write_data restart_relaxation_%i.data nocoeff" % (reset_timestep+transcription_based_loop_extrusion_dynamics['keep_restart_step']))

                reset_timestep=reset_timestep+transcription_based_loop_extrusion_dynamics['keep_restart_step']
                remaining_time=transcription_based_loop_extrusion_dynamics['timesteps_relaxation']-reset_timestep
                print(remaining_time, transcription_based_loop_extrusion_dynamics['timesteps_relaxation'], reset_timestep)
                if remaining_time > 0:
                    print("reset_timestep %d" % reset_timestep)
                    print("run %i" % remaining_time)
                    lmp.command("reset_timestep %d" % reset_timestep)
                    lmp.command("run %i" % remaining_time)
                    #lmp.command("write_restart restart_relaxation_%i.restart" % (reset_timestep+remaining_time))
                    lmp.command("write_data restart_relaxation_%i.data nocoeff" % (reset_timestep+remaining_time))

                print("Finished relaxation %d" % (reset_timestep+remaining_time))
            else:
                print("Relaxation already done: %d" % (initial_reset_timestep))

        # Start Loop extrusion dynamics
        if to_dump:
            lmp.command("undump 1")
            lmp.command("dump    1       all    custom    %i   loop_extrusion_MD_*.XYZ  id  xu yu zu" % to_dump)
            lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id append no")

        print("#Starting loop extrusion dynamics")
        # List of target loops of the form [(loop1_start,loop1_stop),...,(loopN_start,loopN_stop)]
        target_loops  = read_target_transcription_based_loops_input(transcription_based_loop_extrusion_dynamics['target_loops_input'],
                                                                    transcription_based_loop_extrusion_dynamics['chrlength'],
                                                                    transcription_based_loop_extrusion_dynamics['perc_enfor_loops'])
        
        # Randomly extract starting point of the extrusion dynamics between start and stop
        if 'restart_initial_loops' in transcription_based_loop_extrusion_dynamics and os.path.exists(transcription_based_loop_extrusion_dynamics['restart_initial_loops']):
            filename = transcription_based_loop_extrusion_dynamics['restart_initial_loops']
            # restart_initial_loops_step_${step}.txt
            restart_loop_extrusion_step_1 = int(filename.split("_")[4].split(".")[0])

            filename = initial_conformation
            restart_loop_extrusion_step = 0
            # restart_loop_extrusion_step_${step}.data
            restart_loop_extrusion_step_2 = int(filename.split("_")[4].split(".")[0])
            print(restart_loop_extrusion_step_1, restart_loop_extrusion_step_2)
            if restart_loop_extrusion_step_1 == restart_loop_extrusion_step_2:
                restart_loop_extrusion_step = restart_loop_extrusion_step_1
                print("Initial loops and initial conformations to restart from loop-extrusion step %d found" % restart_loop_extrusion_step)
                print("WARNING: It is responsability of the user that the file %s is combatible with the provided %s and %s" % (transcription_based_loop_extrusion_dynamics['target_loops_input'], transcription_based_loop_extrusion_dynamics['restart_initial_loops'], initial_conformation))
            else: 
                print("ERROR: Initial loops file requires to restart from loop-extrusion step %d found" % restart_loop_extrusion_step_1)
                print("but, the initial conformation requires to restart from loop-extrusion step %d found" % restart_loop_extrusion_step_2)
                exit(1)

            initial_loops_tmp = read_target_transcription_based_loops_input(transcription_based_loop_extrusion_dynamics['restart_initial_loops'],
                                                                        transcription_based_loop_extrusion_dynamics['chrlength'],
                                                                        100)
            initial_loops = []
            for initial_loop in initial_loops_tmp:
                print(initial_loop)
                initial_loops.append([initial_loop[0],initial_loop[1]])
                print(initial_loops[-1])
        else:
            restart_loop_extrusion_step = 0
            #initial_loops = draw_loop_extrusion_from_TSS_starting_points(target_loops,
            initial_loops = central_loop_extrusion_starting_points(target_loops,
                                                                transcription_based_loop_extrusion_dynamics['chrlength'])
        print(initial_loops)
        initial_loops_stable = copy.deepcopy(initial_loops)
        #print target_loops,initial_loops

        print("The time for an extrusion event (that may extrude 1 or 2 beads) is %d" % (transcription_based_loop_extrusion_dynamics['timesteps_per_loop_extrusion_step']))

        # Maximum number of particles to be extruded during an extrusion and Maximum number of extrusions
        maximum_number_of_extruded_particles= get_maximum_number_of_extruded_particles(target_loops,initial_loops)
        maximum_number_of_extrusions        = get_maximum_number_of_extrusions(target_loops)

        # Minimum simulation time need to extrude all the requested particles and form the target pairs
        minimum_time_needed_for_requested_extrusion = maximum_number_of_extruded_particles*maximum_number_of_extrusions*transcription_based_loop_extrusion_dynamics['timesteps_per_loop_extrusion_step']
        print("The minimum simulation time to fulfill the all the target pairs by loop extrusion is %d" % (minimum_time_needed_for_requested_extrusion))

        total_run_duration = transcription_based_loop_extrusion_dynamics['max_extrusion_run_duration']
        if transcription_based_loop_extrusion_dynamics['max_extrusion_run_duration'] <= minimum_time_needed_for_requested_extrusion:
            print("WARNING! Increased the transcription_based_loop_extrusion_dynamics['max_extrusion_run_duration'] to %s" % minimum_time_needed_for_requested_extrusion)
            transcription_based_loop_extrusion_dynamics['max_extrusion_run_duration'] = minimum_time_needed_for_requested_extrusion
            total_run_duration = transcription_based_loop_extrusion_dynamics['max_extrusion_run_duration']

        total_extrusion_steps = maximum_number_of_extrusions
        #transcription_based_loop_extrusion_dynamics['max_extrusion_run_duration'] = minimum_time_needed_for_requested_extrusion

        lmp.command("compute xu all property/atom xu")
        lmp.command("compute yu all property/atom yu")
        lmp.command("compute zu all property/atom zu")

        fixed_extruder_number= 0
        thermo_style_fixed   = ""
        try:
            fixed_extruders = transcription_based_loop_extrusion_dynamics['fixed_extruders']
            print("Define the fixed extruders once for all")
            print(fixed_extruders)
            for particle1,particle2 in fixed_extruders:
                fixed_extruder_number += 1
                print("# fix fixed_LE%i all restrain bond %i  %i %f %f %f" % (fixed_extruder_number,
                                                                              particle1,
                                                                              particle2,
                                                                              transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
                                                                              transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
                                                                              transcription_based_loop_extrusion_dynamics['equilibrium_distance']))

                lmp.command("fix fixed_LE%i all restrain bond %i  %i %f %f %f" % (fixed_extruder_number,
                                                                                  particle1,
                                                                                  particle2,
                                                                                  transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
                                                                                  transcription_based_loop_extrusion_dynamics['FE_attraction_strength'],
                                                                                  transcription_based_loop_extrusion_dynamics['equilibrium_distance']))
               
                lmp.command("variable x%i equal c_xu[%i]" % (particle1, particle1))
                lmp.command("variable x%i equal c_xu[%i]" % (particle2, particle2))
                lmp.command("variable y%i equal c_yu[%i]" % (particle1, particle1))
                lmp.command("variable y%i equal c_yu[%i]" % (particle2, particle2))
                lmp.command("variable z%i equal c_zu[%i]" % (particle1, particle1))
                lmp.command("variable z%i equal c_zu[%i]" % (particle2, particle2))

                lmp.command("variable xfixedLE%i equal v_x%i-v_x%i" % (fixed_extruder_number,
                                                                  particle1,
                                                                  particle2))
                lmp.command("variable yfixedLE%i equal v_y%i-v_y%i" % (fixed_extruder_number,
                                                                  particle1,
                                                                  particle2))
                lmp.command("variable zfixedLE%i equal v_z%i-v_z%i" % (fixed_extruder_number,
                                                                  particle1,
                                                                  particle2))
                lmp.command("variable Fdist_%i_%i equal sqrt(v_xfixedLE%i*v_xfixedLE%i+v_yfixedLE%i*v_yfixedLE%i+v_zfixedLE%i*v_zfixedLE%i)" % (particle1,
                                                                                                                                               particle2,
                                                                                                                                               fixed_extruder_number,
                                                                                                                                               fixed_extruder_number,
                                                                                                                                               fixed_extruder_number,
                                                                                                                                               fixed_extruder_number,
                                                                                                                                               fixed_extruder_number,
                                                                                                                                               fixed_extruder_number))

                thermo_style_fixed += " v_Fdist_%i_%i" % (particle1,
                                                         particle2)
        except:
            pass
        print("Defined",fixed_extruder_number,"fixed extruders")
        print(thermo_style_fixed)
        
        # Loop extrusion steps
        if restart_loop_extrusion_step > 0:
            print("reset_timestep %d" % (transcription_based_loop_extrusion_dynamics['timesteps_per_loop_extrusion_step']*(restart_loop_extrusion_step-1)))
            lmp.command("reset_timestep %d" % (transcription_based_loop_extrusion_dynamics['timesteps_per_loop_extrusion_step']*(restart_loop_extrusion_step-1)))
        else:
            print("reset_timestep 0")
            lmp.command("reset_timestep 0")
        for extrusion in range(total_extrusion_steps):

            print("#Extrusion round",extrusion+restart_loop_extrusion_step+1)

            initial_loops_tmp = []
            initial_loops = copy.deepcopy(initial_loops_stable)

            for iloop in range(len(initial_loops)):
                print("#Loop ",iloop+1," extruded ",target_loops[iloop][2]," times")
                if target_loops[iloop][2] >  extrusion:
                    initial_loops_tmp.append(initial_loops[iloop])
                else:
                    initial_loops_tmp.append(target_loops[iloop][0:2])
            #print extrusion,"Initial loops: ",initial_loops_tmp

            lmp.command("compute xu all property/atom xu")
            lmp.command("compute yu all property/atom yu")
            lmp.command("compute zu all property/atom zu")

            for LES in range(1,maximum_number_of_extruded_particles+1):
                thermo_style="thermo_style   custom   step temp epair emol"
                thermo_style += thermo_style_fixed

                # Loop extrusion steps
                loop_number = 1
                for particle1,particle2 in initial_loops_tmp:
                    if abs(particle1-particle2) > 1:
                        

                        print("# fix LE%i all restrain bond %i  %i %f %f %f %f" % (loop_number,
                                                                                   particle1,
                                                                                   particle2,
                                                                                   0.0,
                                                                                   transcription_based_loop_extrusion_dynamics['attraction_strength'],
                                                                                   2.0,
                                                                                   transcription_based_loop_extrusion_dynamics['equilibrium_distance']))
                    
                        lmp.command("fix LE%i all restrain bond %i  %i %f %f %f %f" % (loop_number,
                                                                                       particle1,
                                                                                       particle2,
                                                                                       0.0,
                                                                                       transcription_based_loop_extrusion_dynamics['attraction_strength'],
                                                                                       2.0,
                                                                                       transcription_based_loop_extrusion_dynamics['equilibrium_distance']))                     
                    lmp.command("variable x%i equal c_xu[%i]" % (particle1, particle1))
                    lmp.command("variable x%i equal c_xu[%i]" % (particle2, particle2))
                    lmp.command("variable y%i equal c_yu[%i]" % (particle1, particle1))
                    lmp.command("variable y%i equal c_yu[%i]" % (particle2, particle2))
                    lmp.command("variable z%i equal c_zu[%i]" % (particle1, particle1))
                    lmp.command("variable z%i equal c_zu[%i]" % (particle2, particle2))

                    lmp.command("variable xLE%i equal v_x%i-v_x%i" % (loop_number,
                                                                      particle1,
                                                                      particle2))
                    lmp.command("variable yLE%i equal v_y%i-v_y%i" % (loop_number,
                                                                      particle1,
                                                                      particle2))
                    lmp.command("variable zLE%i equal v_z%i-v_z%i" % (loop_number,
                                                                      particle1,
                                                                      particle2))
                    lmp.command("variable dist_%i_%i equal sqrt(v_xLE%i*v_xLE%i+v_yLE%i*v_yLE%i+v_zLE%i*v_zLE%i)" % (particle1,
                                                                                                                     particle2,
                                                                                                                     loop_number,
                                                                                                                     loop_number,
                                                                                                                     loop_number,
                                                                                                                     loop_number,
                                                                                                                     loop_number,
                                                                                                                     loop_number))
                    thermo_style += " v_dist_%i_%i" % (particle1,
                                                       particle2)
                    loop_number += 1

                # Doing the LES
                lmp.command("%s" % thermo_style)
                runtime = transcription_based_loop_extrusion_dynamics['timesteps_per_loop_extrusion_step']
                print("run %i" % runtime)
                lmp.command("run %i" % runtime)

                # Remove the loop extrusion restraint!
                loop_number = 1
                for particle1,particle2 in initial_loops_tmp:
                    if abs(particle1-particle2) > 1:
                        print("# unfix LE%i" % (loop_number))
                        lmp.command("unfix LE%i" % (loop_number))                        
                        loop_number += 1

                if LES == maximum_number_of_extruded_particles:
                    for particle1,particle2 in initial_loops_tmp:
                        if abs(particle1-particle2) > 1:
                            
                            
                            print("# fix LE%i all restrain bond %i  %i %f %f %f %f" % (loop_number,
                                                                                       particle1,
                                                                                       particle2,
                                                                                       transcription_based_loop_extrusion_dynamics['attraction_strength'],
                                                                                       transcription_based_loop_extrusion_dynamics['attraction_strength'],
                                                                                       transcription_based_loop_extrusion_dynamics['equilibrium_distance'],
                                                                                       transcription_based_loop_extrusion_dynamics['equilibrium_distance']))
                            
                            lmp.command("fix LE%i all restrain bond %i  %i %f %f %f %f" % (loop_number,
                                                                                           particle1,
                                                                                           particle2,
                                                                                           transcription_based_loop_extrusion_dynamics['attraction_strength'],
                                                                                           transcription_based_loop_extrusion_dynamics['attraction_strength'],
                                                                                           transcription_based_loop_extrusion_dynamics['equilibrium_distance'],
                                                                                           transcription_based_loop_extrusion_dynamics['equilibrium_distance']))                     
                            lmp.command("variable x%i equal c_xu[%i]" % (particle1, particle1))
                            lmp.command("variable x%i equal c_xu[%i]" % (particle2, particle2))
                            lmp.command("variable y%i equal c_yu[%i]" % (particle1, particle1))
                            lmp.command("variable y%i equal c_yu[%i]" % (particle2, particle2))
                            lmp.command("variable z%i equal c_zu[%i]" % (particle1, particle1))
                            lmp.command("variable z%i equal c_zu[%i]" % (particle2, particle2))
                            
                            lmp.command("variable xLE%i equal v_x%i-v_x%i" % (loop_number,
                                                                              particle1,
                                                                              particle2))
                            lmp.command("variable yLE%i equal v_y%i-v_y%i" % (loop_number,
                                                                              particle1,
                                                                              particle2))
                            lmp.command("variable zLE%i equal v_z%i-v_z%i" % (loop_number,
                                                                              particle1,
                                                                              particle2))
                            lmp.command("variable dist_%i_%i equal sqrt(v_xLE%i*v_xLE%i+v_yLE%i*v_yLE%i+v_zLE%i*v_zLE%i)" % (particle1,
                                                                                                                             particle2,
                                                                                                                             loop_number,
                                                                                                                             loop_number,
                                                                                                                             loop_number,
                                                                                                                             loop_number,
                                                                                                                             loop_number,
                                                                                                                             loop_number))
                            thermo_style += " v_dist_%i_%i" % (particle1,
                                                               particle2)
                            loop_number += 1
                    
                    runtime = total_run_duration - (transcription_based_loop_extrusion_dynamics['timesteps_per_loop_extrusion_step']*(maximum_number_of_extruded_particles))
                    lmp.command("run %i" % runtime)

                # Update the particles involved in the loop extrusion interaction:
                # decrease the initial_start by one until you get to start
                # increase the initial_stop by one until you get to stop
                for initial_loop_tmp, target_loop in zip(initial_loops_tmp,target_loops):

                    if initial_loop_tmp[0] > target_loop[0]:
                        initial_loop_tmp[0] -= 1
                    if initial_loop_tmp[1] < target_loop[1]:
                        initial_loop_tmp[1] += 1


                # Save the restart for the next step
                # restart_loop_extrusion_step_${step}.txt
                #lmp.command("write_restart restart_loop_extrusion_step_%i.restart" % (LES+restart_loop_extrusion_step))
                #print"write_restart restart_loop_extrusion_step_%i.restart" % (LES+restart_loop_extrusion_step)
                print("write_data restart_loop_extrusion_step_%d.data nocoeff" % (LES+restart_loop_extrusion_step))
                lmp.command("write_data restart_loop_extrusion_step_%d.data nocoeff" % (LES+restart_loop_extrusion_step))
                # Save the loops to impose at the next step
                # restart_initial_loops_step_${step}.txt
                tmp_out = open("restart_initial_loops_step_%d.txt" % (LES+restart_loop_extrusion_step), "w")
                for initial_loop_tmp in initial_loops_tmp:
                    # chr19 58212 582281
                    print("chr %d %d 1" % (initial_loop_tmp[0],initial_loop_tmp[1]))
                    tmp_out.write("chr %d %d 1\n" % (initial_loop_tmp[0],initial_loop_tmp[1]))

    

    #if to_dump:
    #    lmp.command("undump 1")
    #    lmp.command("dump    1       all    custom    %i   langevin_dynamics_*.XYZ  id  xu yu zu" % to_dump)
    #    lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id append yes")
            
    timepoints = 1
    xc = []
    # Setup the pairs to co-localize using the COLVARS plug-in
    if steering_pairs:
        
        if doRestart == False:
            # Start relaxation step
            lmp.command("reset_timestep 0")   # cambiar para punto ionicial
            lmp.command("run %i" % steering_pairs['timesteps_relaxation'])
            lmp.command("reset_timestep %i" % 0)
        
            # Start Steered Langevin dynamics
            if to_dump:
                lmp.command("undump 1")
                lmp.command("dump    1       all    custom    %i   %ssteered_MD_*.XYZ  id  xu yu zu" % (to_dump,lammps_folder))
                #lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id")

        if 'number_of_kincrease' in steering_pairs:
            nbr_kincr = steering_pairs['number_of_kincrease']
        else:
            nbr_kincr = 1
        
        if doRestart == True:
            restart_k_increase = int(restart_file.split('/')[-1].split('_')[2])
            restart_time       = int(restart_file.split('/')[-1].split('_')[4][:-8])

        #steering_pairs['colvar_output'] = os.path.dirname(os.path.abspath(steering_pairs['colvar_output'])) + '/' + str(kseed) + '_'+ os.path.basename(steering_pairs['colvar_output'])    
        steering_pairs['colvar_output'] = lammps_folder+os.path.basename(steering_pairs['colvar_output'])
        for kincrease in range(nbr_kincr):
            # Write the file containing the pairs to constraint
            # steering_pairs should be a dictionary with:
            # Avoid to repeat calculations in case of restart
            if (doRestart == True) and (kincrease < restart_k_increase):
                continue

            if useColvars == True:
                
                generate_colvars_list(steering_pairs, kincrease+1)

                # Adding the colvar option
                #print "fix 4 all colvars %s output %s" % (steering_pairs['colvar_output'],lammps_folder)
                lmp.command("fix 4 all colvars %s output %sout" % (steering_pairs['colvar_output'],lammps_folder))

                if to_dump:
                    # lmp.command("thermo_style   custom   step temp epair emol")
                    lmp.command("thermo_style   custom   step temp epair emol pe ke etotal f_4")
                    lmp.command("thermo_modify norm no flush yes")
                    lmp.command("variable step equal step")
                    lmp.command("variable objfun equal f_4")
                    lmp.command('''fix 5 all print %s "${step} ${objfun}" file "%sobj_fun_from_time_point_%s_to_time_point_%s.txt" screen "no" title "#Timestep Objective_Function"''' % (steering_pairs['colvar_dump_freq'],lammps_folder,str(0), str(1)))

            # will load the bonds directly into LAMMPS
            else:
                bond_list = generate_bond_list(steering_pairs)
                for bond in bond_list:
                    lmp.command(bond)

                if to_dump:
                    lmp.command("thermo_style   custom   step temp etotal")
                    lmp.command("thermo_modify norm no flush yes")
                    lmp.command("variable step equal step")
                    lmp.command("variable objfun equal etotal")
                    lmp.command('''fix 5 all print %s "${step} ${objfun}" file "%sobj_fun_from_time_point_%s_to_time_point_%s.txt" screen "no" title "#Timestep Objective_Function"''' % (steering_pairs['colvar_dump_freq'],lammps_folder,str(0), str(1)))
            #lmp.command("reset_timestep %i" % 0)
            resettime = 0
            runtime   = steering_pairs['timesteps_per_k']
            if (doRestart == True) and (kincrease == restart_k_increase):
                resettime = restart_time 
                runtime   = steering_pairs['timesteps_per_k'] - restart_time

            # Create 10 restarts with name restart_kincrease_%s_time_%s.restart every
            if saveRestart == True:
                if os.path.isdir(restart_file):
                    restart_file_new = restart_file + 'restart_kincrease_%s_time_*.restart' %(kincrease)
                else:
                    restart_file_new = '/'.join(restart_file.split('/')[:-1]) + '/restart_kincrease_%s_time_*.restart' %(kincrease)
                #print(restart_file_new)
                lmp.command("restart %i %s" %(int(steering_pairs['timesteps_per_k']/store_n_steps), restart_file_new))

            #lmp.command("reset_timestep %i" % resettime)
            lmp.command("run %i" % runtime)

    # Setup the pairs to co-localize using the COLVARS plug-in
    if time_dependent_steering_pairs:
        timepoints = time_dependent_steering_pairs['colvar_dump_freq']

        #if exists("objective_function_profile.txt"):
        #    os.remove("objective_function_profile.txt")

        #print "# Getting the time dependent steering pairs!"
        time_dependent_restraints = get_time_dependent_colvars_list(time_dependent_steering_pairs)
        time_points = sorted(time_dependent_restraints.keys())
        print("#Time_points",time_points)        
        sys.stdout.flush()            

        time_dependent_steering_pairs['colvar_output'] = lammps_folder+os.path.basename(time_dependent_steering_pairs['colvar_output'])
        # Performing the adaptation step from initial conformation to Tadphys excluded volume
        if time_dependent_steering_pairs['adaptation_step']:
            restraints = {}
            for time_point in time_points[0:1]:
                lmp.command("reset_timestep %i" % 0)    
                # Change to_dump with some way to load the conformations you want to store
                # This Adaptation could be discarded in the trajectory files.
                if to_dump:
                    lmp.command("undump 1")
                    lmp.command("dump    1       all    custom    %i  %sadapting_MD_from_initial_conformation_to_Tadphys_at_time_point_%s.XYZ  id  xu yu zu" % (to_dump, lammps_folder, time_point))
                    lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id append yes")

                restraints[time_point] = {}
                print("# Step %s - %s" % (time_point, time_point))
                sys.stdout.flush()            
                for pair in list(time_dependent_restraints[time_point].keys()):
                    # Strategy changing gradually the spring constant and the equilibrium distance
                    # Case 1: The restraint is present at time point 0 and time point 1:
                    if pair in time_dependent_restraints[time_point]:
                        # Case A: The restrainttype is the same at time point 0 and time point 1 ->
                        # The spring force changes, and the equilibrium distance is the one at time_point+1
                        restraints[time_point][pair] = [
                            # Restraint type
                            [time_dependent_restraints[time_point][pair][0]], 
                            # Initial spring constant 
                            [time_dependent_restraints[time_point][pair][1]*time_dependent_steering_pairs['k_factor']], 
                            # Final spring constant 
                            [time_dependent_restraints[time_point][pair][1]*time_dependent_steering_pairs['k_factor']], 
                            # Initial equilibrium distance
                            [time_dependent_restraints[time_point][pair][2]], 
                            # Final equilibrium distance
                            [time_dependent_restraints[time_point][pair][2]], 
                            # Number of timesteps for the gradual change
                            [int(time_dependent_steering_pairs['timesteps_per_k_change'][time_point]*0.1)]]
                if useColvars == True:
                    generate_time_dependent_colvars_list(restraints[time_point], time_dependent_steering_pairs['colvar_output'], time_dependent_steering_pairs['colvar_dump_freq'])
                    copyfile(time_dependent_steering_pairs['colvar_output'], 
                             "colvar_list_from_time_point_%s_to_time_point_%s.txt" % 
                             (str(time_point), str(time_point)))
                    lmp.command("velocity all create 1.0 %s" % randint(1,100000))
                    # Adding the colvar option and perfoming the steering
                    if time_point != time_points[0]:
                        lmp.command("unfix 4")
                    print("#fix 4 all colvars %s" % time_dependent_steering_pairs['colvar_output'])
                    sys.stdout.flush()
                    lmp.command("fix 4 all colvars %s tstat 2 output %sout" % (time_dependent_steering_pairs['colvar_output'],lammps_folder))
                else:
                    bond_list = generate_time_dependent_bond_list(restraints[time_point])
                    for bond in bond_list:
                        lmp.command(bond)

                lmp.command("run %i" % int(time_dependent_steering_pairs['timesteps_per_k_change'][time_point]*0.1))

        # Time dependent steering
        restraints = {}
        #for i in range(time_points[0],time_points[-1]):
        for time_point in time_points[0:-1]:
            lmp.command("reset_timestep %i" % 0)    
            # Change to_dump with some way to load the conformations you want to store
            if to_dump:
                lmp.command("undump 1")
                lmp.command("dump    1       all    custom    %i   %ssteered_MD_from_time_point_%s_to_time_point_%s.XYZ  id  xu yu zu" % (to_dump, lammps_folder, time_point, time_point+1))
                lmp.command("dump_modify     1 format line \"%d %.5f %.5f %.5f\" sort id append yes")

            restraints[time_point] = {}
            print("# Step %s - %s" % (time_point, time_point+1))
            sys.stdout.flush()            
            # Compute the current distance between any two particles
            xc_tmp = np.array(lmp.gather_atoms("xu",1,3))                
            #xc_tmp = np.array(lmp.extract_compute("ucoordinates",1,3))
            current_distances = compute_particles_distance(xc_tmp)

            for pair in set(list(time_dependent_restraints[time_point].keys())+list(time_dependent_restraints[time_point+1].keys())):                
                r = 0
                
                # Strategy changing gradually the spring constant
                # Case 1: The restraint is present at time point 0 and time point 1:
                if pair     in time_dependent_restraints[time_point] and pair     in time_dependent_restraints[time_point+1]:
                    # Case A: The restrainttype is the same at time point 0 and time point 1 ->
                    # The spring force changes, and the equilibrium distance is the one at time_point+1
                    if time_dependent_restraints[time_point][pair][0]   == time_dependent_restraints[time_point+1][pair][0]:
                        r += 1
                        restraints[time_point][pair] = [
                            # Restraint type
                            [time_dependent_restraints[time_point+1][pair][0]], 
                            # Initial spring constant 
                            [time_dependent_restraints[time_point][pair][1]  *time_dependent_steering_pairs['k_factor']], 
                            # Final spring constant 
                            [time_dependent_restraints[time_point+1][pair][1]*time_dependent_steering_pairs['k_factor']], 
                            # Initial equilibrium distance
                            [time_dependent_restraints[time_point][pair][2]], 
                            # Final equilibrium distance
                            [time_dependent_restraints[time_point+1][pair][2]], 
                            # Number of timesteps for the gradual change
                            [int(time_dependent_steering_pairs['timesteps_per_k_change'][time_point])]]
                    # Case B: The restrainttype is different between time point 0 and time point 1
                    if time_dependent_restraints[time_point][pair][0]   != time_dependent_restraints[time_point+1][pair][0]:
                        # Case a: The restrainttype is "Harmonic" at time point 0 
                        # and "LowerBoundHarmonic" at time point 1                        
                        if time_dependent_restraints[time_point][pair][0] == "Harmonic":
                            r += 1
                            restraints[time_point][pair] = [
                                # Restraint type
                                [time_dependent_restraints[time_point][pair][0], time_dependent_restraints[time_point+1][pair][0]], 
                                # Initial spring constant 
                                [time_dependent_restraints[time_point][pair][1]*time_dependent_steering_pairs['k_factor'], 0.0],
                                # Final spring constant 
                                [0.0, time_dependent_restraints[time_point+1][pair][1]*time_dependent_steering_pairs['k_factor']],
                                # Initial equilibrium distance
                                [time_dependent_restraints[time_point][pair][2], time_dependent_restraints[time_point][pair][2]],
                                # Final equilibrium distance
                                [time_dependent_restraints[time_point+1][pair][2], time_dependent_restraints[time_point+1][pair][2]],
                                # Number of timesteps for the gradual change
                                #[int(time_dependent_steering_pairs['timesteps_per_k_change']), int(time_dependent_steering_pairs['timesteps_per_k_change'])]]
                                [int(time_dependent_steering_pairs['timesteps_per_k_change'][time_point]), int(time_dependent_steering_pairs['timesteps_per_k_change'][time_point])]]
                        # Case b: The restrainttype is "LowerBoundHarmonic" at time point 0 
                        # and "Harmonic" at time point 1
                        if time_dependent_restraints[time_point][pair][0] == "HarmonicLowerBound":
                            r += 1
                            restraints[time_point][pair] = [
                                # Restraint type
                                [time_dependent_restraints[time_point][pair][0], time_dependent_restraints[time_point+1][pair][0]], 
                                # Initial spring constant 
                                [time_dependent_restraints[time_point][pair][1]*time_dependent_steering_pairs['k_factor'], 0.0],
                                # Final spring constant 
                                [0.0, time_dependent_restraints[time_point+1][pair][1]*time_dependent_steering_pairs['k_factor']],
                                # Initial equilibrium distance
                                [time_dependent_restraints[time_point][pair][2], time_dependent_restraints[time_point][pair][2]],
                                # Final equilibrium distance
                                [time_dependent_restraints[time_point+1][pair][2], time_dependent_restraints[time_point+1][pair][2]],
                                # Number of timesteps for the gradual change
                                #[int(time_dependent_steering_pairs['timesteps_per_k_change']), int(time_dependent_steering_pairs['timesteps_per_k_change'])]]
                                [int(time_dependent_steering_pairs['timesteps_per_k_change'][time_point]), int(time_dependent_steering_pairs['timesteps_per_k_change'][time_point])]]

                # Case 2: The restraint is not present at time point 0, but it is at time point 1:                            
                elif pair not in time_dependent_restraints[time_point] and pair     in time_dependent_restraints[time_point+1]:
                    # List content: restraint_type,kforce,distance
                    r += 1
                    restraints[time_point][pair] = [
                        # Restraint type -> Is the one at time point time_point+1
                        [time_dependent_restraints[time_point+1][pair][0]],
                        # Initial spring constant 
                        [0.0],
                        # Final spring constant 
                        [time_dependent_restraints[time_point+1][pair][1]*time_dependent_steering_pairs['k_factor']], 
                        # Initial equilibrium distance 
                        [time_dependent_restraints[time_point+1][pair][2]], 
                        # Final equilibrium distance 
                        [time_dependent_restraints[time_point+1][pair][2]], 
                        # Number of timesteps for the gradual change
                        [int(time_dependent_steering_pairs['timesteps_per_k_change'][time_point])]] 

                # Case 3: The restraint is     present at time point 0, but it is not at time point 1:                            
                elif pair     in time_dependent_restraints[time_point] and pair not in time_dependent_restraints[time_point+1]:
                    # List content: restraint_type,kforce,distance
                    r += 1
                    restraints[time_point][pair] = [
                        # Restraint type -> Is the one at time point time_point
                        [time_dependent_restraints[time_point][pair][0]], 
                        # Initial spring constant 
                        [time_dependent_restraints[time_point][pair][1]*time_dependent_steering_pairs['k_factor']],                         
                        # Final spring constant 
                        [0.0],
                        # Initial equilibrium distance 
                        [time_dependent_restraints[time_point][pair][2]],                         
                        # Final equilibrium distance 
                        [time_dependent_restraints[time_point][pair][2]], 
                        # Number of timesteps for the gradual change
                        [int(time_dependent_steering_pairs['timesteps_per_k_change'][time_point])]]
                
                    #current_distances[pair],                          
                else:
                    print("#ERROR None of the previous conditions is matched!")
                    if pair     in time_dependent_restraints[time_point]:
                        print("# Pair %s at timepoint %s %s  " % (pair, time_point, time_dependent_restraints[time_point][pair]))
                    if pair     in time_dependent_restraints[time_point+1]:
                        print("# Pair %s at timepoint %s %s  " % (pair, time_point+1, time_dependent_restraints[time_point+1][pair]))
                    continue

                if r > 1:
                    print("#ERROR Two of the previous conditions are matched!")

                #if pair     in time_dependent_restraints[time_point]:
                #    print "# Pair %s at timepoint %s %s  " % (pair, time_point, time_dependent_restraints[time_point][pair])
                #else:
                #    print "# Pair %s at timepoint %s None" % (pair, time_point)

                #if pair     in time_dependent_restraints[time_point+1]:
                #    print "# Pair %s at timepoint %s %s  " % (pair, time_point+1, time_dependent_restraints[time_point+1][pair])
                #else:
                #    print "# Pair %s at timepoint %s None" % (pair, time_point+1)
                #print restraints[pair]
                #print ""
            lmp.command("velocity all create 1.0 %s" % randint(1,100000))
            if useColvars == True:
                generate_time_dependent_colvars_list(restraints[time_point], time_dependent_steering_pairs['colvar_output'], time_dependent_steering_pairs['colvar_dump_freq'])
                copyfile(time_dependent_steering_pairs['colvar_output'], 
                         "%scolvar_list_from_time_point_%s_to_time_point_%s.txt" % 
                         (lammps_folder, str(time_point), str(time_point+1)))
                # Adding the colvar option and perfoming the steering
                if time_point != time_points[0]:
                    lmp.command("unfix 4")
                print("#fix 4 all colvars %s" % time_dependent_steering_pairs['colvar_output'])
                sys.stdout.flush()
                lmp.command("fix 4 all colvars %s tstat 2 output %sout" % (time_dependent_steering_pairs['colvar_output'],lammps_folder))
                if to_dump:
                    lmp.command("thermo_style   custom   step temp epair emol pe ke etotal f_4")
                    lmp.command("thermo_modify norm no flush yes")
                    lmp.command("variable step equal step")
                    lmp.command("variable objfun equal f_4")
                    lmp.command('''fix 5 all print %s "${step} ${objfun}" file "%sobj_fun_from_time_point_%s_to_time_point_%s.txt" screen "no" title "#Timestep Objective_Function"''' % (time_dependent_steering_pairs['colvar_dump_freq'],lammps_folder,str(time_point), str(time_point+1)))
            else:
                bond_list = generate_time_dependent_bond_list(restraints[time_point])
                for bond in bond_list:
                    lmp.command(bond)
                if to_dump:
                    lmp.command("thermo_style   custom   step temp epair emol pe ke etotal")
                    lmp.command("thermo_modify norm no flush yes")
                    lmp.command("variable step equal step")
                    lmp.command("variable objfun equal etotal")
                    lmp.command('''fix 5 all print %s "${step} ${objfun}" file "%sobj_fun_from_time_point_%s_to_time_point_%s.txt" screen "no" title "#Timestep Objective_Function"''' % (time_dependent_steering_pairs['colvar_dump_freq'],lammps_folder,str(time_point), str(time_point+1)))
            
            lmp.command("run %i" % int(time_dependent_steering_pairs['timesteps_per_k_change'][time_point]))
            
            if time_point > 0:
                    
                if exists("%sout.colvars.traj.BAK" % lammps_folder):

                    copyfile("%sout.colvars.traj.BAK" % lammps_folder, "%srestrained_pairs_equilibrium_distance_vs_timestep_from_time_point_%s_to_time_point_%s.txt" % (lammps_folder, str(time_point-1), str(time_point)))
            
                    os.remove("%sout.colvars.traj.BAK" % lammps_folder)


    #print(compartmentalization)
    # Set interactions for chromosome compartmentalization
    if compartmentalization:

        if 'gyration' in compartmentalization:
            lmp.command("compute RgSquared all gyration")
            #print("compute RgSquared all gyration")
            lmp.command("variable RgSquared equal c_RgSquared")
            #lmp.command("variable RgSquaredxx equal c_RgSquared[1]")
            #lmp.command("variable RgSquaredyy equal c_RgSquared[2]")
            #lmp.command("variable RgSquaredzz equal c_RgSquared[3]")
            lmp.command("variable step equal step")
            #lmp.command("fix RgSquared all print %i \"${step} ${RgSquared} ${RgSquaredxx} ${RgSquaredyy} ${RgSquaredzz}\" file %sRg.txt" % (compartmentalization['gyration'],lammps_folder))
            lmp.command("fix RgSquared all print %i \"${step} ${RgSquared}\" file %sRg.txt" % (compartmentalization['gyration'],lammps_folder))
            #lmp.command("thermo_style   custom   step temp epair emol pe ke etotal c_RgSquared")
            #print("thermo_style   custom   step temp epair emol pe ke etotal c_RgSquared")
            #lmp.command("thermo_modify norm no flush yes")
            #print("thermo_modify norm no flush yes")

        if 'Ree' in compartmentalization:
            lmp.command("compute xu all property/atom xu")
            lmp.command("compute yu all property/atom yu")
            lmp.command("compute zu all property/atom zu")
            lmp.command("variable Reex equal c_xu[1]-c_xu[%i]" % (chromosome_particle_numbers[0]))
            lmp.command("variable Reey equal c_yu[1]-c_yu[%i]" % (chromosome_particle_numbers[0]))
            lmp.command("variable Reez equal c_zu[1]-c_zu[%i]" % (chromosome_particle_numbers[0]))
            lmp.command("variable step equal step")
            lmp.command("fix Ree all print %i \"${step} ${Reex} ${Reey} ${Reez}\" file %sRee.txt" % (compartmentalization['Ree'],lammps_folder))
            
        if 'distances' in compartmentalization:
            lmp.command("compute pairs     all property/local patom1 patom2")
            #print("compute pairs     all property/local patom1 patom2")
            lmp.command("compute distances all pair/local dist")
            #print("compute distances all pair/local dist")
            lmp.command("dump distances all local %i %sdistances_*.txt c_pairs[1] c_pairs[2] c_distances" % (compartmentalization['distances'],lammps_folder))
            #print("dump distances all local %i %sdistances_*.txt c_pairs[1] c_pairs[2] c_distances" % (compartmentalization['distances'],lammps_folder))
            lmp.command("dump_modify distances delay 1")
            #print("dump_modify distances delay 1")

            
        if to_dump:
            lmp.command("undump 1")
            lmp.command("dump    1       all    custom    %i   %scompartmentalization_*.XYZ  id  xu yu zu" % (to_dump,lammps_folder))
            lmp.command("dump_modify 1 format line \"%d %.5f %.5f %.5f\" sort id append no")

            
        # First we have to partition the genome in the defined compartments
        for group in compartmentalization['partition']:

            frac = 1.0
            if 'fraction' in compartmentalization:
                frac = compartmentalization['fraction'][int(group)]
            
            # Check that group is involved in at least one interaction
            active = 0
            for pair in compartmentalization['interactions']:
                t1 = pair[0]
                t2 = pair[1]
                if t1 == int(group) or t2 == int(group):
                    epsilon = compartmentalization['interactions'][pair][1]
                    if epsilon != 0.0:
                        active = 1

            if active == 1:
                if isinstance(compartmentalization['partition'][group], (int)):
                    compartmentalization['partition'][group] = [compartmentalization['partition'][group]]
                list_of_particles = get_list(compartmentalization['partition'][group])
            
                for atom in list_of_particles:
                    random_number = uniform(0, 1)
                    if random_number <= frac:
                        #print("set atom %s type %s" % (atom,group+1))
                        lmp.command("set atom %s type %s" % (atom,group+1))
                    
        lmp.command("pair_coeff * * lj/cut 1 1.0 1.0 1.12246152962189")
        lmp.command("pair_coeff * * lj/cut 2 1.0 1.0 1.12246152962189")        
        # Second we have to define the type of interactions
        for pair in compartmentalization['interactions']:
            #pair_coeff t1 t2 epsilon sigma rc
            t1 = pair[0]+1
            t2 = pair[1]+1 
            if t1 > t2:
                t1 = pair[1]+1
                t2 = pair[0]+1 

            epsilon = compartmentalization['interactions'][pair][1]

            try:
                sigma1 = compartmentalization['radii'][pair[0]]
            except:
                sigma1 = 0.5
            try: 
                sigma2 = compartmentalization['radii'][pair[1]]
            except:
                sigma2 = 0.5
            sigma = sigma1 + sigma2
                
            if compartmentalization['interactions'][pair][0] == "attraction":
                rc = sigma * 2.5
            if compartmentalization['interactions'][pair][0] == "repulsion":
                sigma = 3.0
                rc    = sigma * 1.12246152962189

            if epsilon != 0.0:
                print("pair_coeff %s %s lj/cut 2 %s %s %s" % (t1,t2,epsilon,sigma,rc))
                lmp.command("pair_coeff %s %s lj/cut 2 %s %s %s" % (t1,t2,epsilon,sigma,rc))

        if 'reset_timestep' in compartmentalization:
            lmp.command("reset_timestep %d" % compartmentalization['reset_timestep'])
                
        try:
            #lmp.command("minimize 1.0e-4 1.0e-6 100000 100000")            
            lmp.command("run %s" % (compartmentalization['runtime']))
            lmp.command("write_data %srestart_conformation.txt nocoeff" % lammps_folder)
        except:
            pass

    # Setup the pairs to co-localize using restraints
    #if restraints:

    #    lmp.command("group chromatin id <> 1 %d" % chromosome_particle_numbers[0])
    #    lmp.command("fix 1 chromatin nve")
    #    lmp.command("fix 2 chromatin langevin 1.0  1.0  %f %i" % (gamma,randint(1,100000)))

    #    restrain_number=0
    #    if restraints["pairs"]:
    #        for particle1,particle2 in restraints["pairs"]:
    #            print("# fix RESTR%i all restrain bond %i  %i %f %f %f" % (restrain_number,
    #                                                                    particle1,
    #                                                                    particle2,
    #                                                                    restraints['attraction_strength'],
    #                                                                    restraints['attraction_strength'],
    #                                                                    restraints['equilibrium_distance']))
            
    #            lmp.command("fix RESTR%i all restrain bond %i  %i %f %f %f" % (restrain_number,
    #                                                                        particle1,
    #                                                                        particle2,
    #                                                                        restraints['attraction_strength'],
    #                                                                        restraints['attraction_strength'],
    #                                                                        restraints['equilibrium_distance']))
    #            lmp.command("variable x%i equal c_xu[%i]" % (particle1, particle1))
    #            lmp.command("variable x%i equal c_xu[%i]" % (particle2, particle2))
    #            lmp.command("variable y%i equal c_yu[%i]" % (particle1, particle1))
    #            lmp.command("variable y%i equal c_yu[%i]" % (particle2, particle2))
    #            lmp.command("variable z%i equal c_zu[%i]" % (particle1, particle1))
    #            lmp.command("variable z%i equal c_zu[%i]" % (particle2, particle2))
            
    #            lmp.command("variable xRESTR%i equal v_x%i-v_x%i" % (restrain_number, particle1, particle2))
    #            lmp.command("variable yRESTR%i equal v_y%i-v_y%i" % (restrain_number, particle1, particle2))
    #            lmp.command("variable zRESTR%i equal v_z%i-v_z%i" % (restrain_number, particle1, particle2))
    #            lmp.command("variable dist_%i_%i equal sqrt(v_xRESTR%i*v_xRESTR%i+v_yRESTR%i*v_yRESTR%i+v_zRESTR%i*v_zRESTR%i)" % (particle1,
    #                                                                                                             particle2,
    #                                                                                                             restrain_number,
    #                                                                                                             restrain_number,
    #                                                                                                             restrain_number,
    #                                                                                                             restrain_number,
    #                                                                                                             restrain_number,
    #                                                                                                             restrain_number))
    #            thermo_style += " v_dist_%i_%i" % (particle1, particle2)
    #            restrain_number += 1
                
    #            lmp.command("%s" % thermo_style)



    
    if loop_extrusion_dynamics_OLD:        
        
        np.random.seed(424242)
        
        print("left_extrusion_rate",loop_extrusion_dynamics["left_extrusion_rate"])
        print("right_extrusion_rate",loop_extrusion_dynamics["right_extrusion_rate"])
        
        if 'lifetimeConstant' in loop_extrusion_dynamics:
            lifetimeExponential = 0
            print("Target extruders lifetimes will be always equal to",loop_extrusion_dynamics['lifetime'])
        else:
            lifetimeExponential = 1
            print("Target extruders lifetimes will be drawn from an exponential distribution with average equal to",loop_extrusion_dynamics['lifetime'])

        natoms = int(lmp.get_natoms())
        print(natoms)
        lmp.command("fix f_unwr all store/state 1 xu yu zu")
        xc_tmp = np.array(lmp.extract_fix("f_unwr",1,2).contents[0:(natoms-2)])
        distances = compute_particles_distance(xc_tmp)
        #print(xc_tmp)
        #print(distances)
        
        #lmp.command("compute forces all pair/local fx fy fz")
        #lmp.command("dump forces all local 1000 pairwise_forces.dump index c_forces[1] c_forces[2] c_forces[3]")

        # Start relaxation step
        try:
            #lmp.command("reset_timestep 0")
            lmp.command("run %i" % loop_extrusion_dynamics['timesteps_relaxation'])
        except:
            pass

        #try :
            #lmp.command("reset_timestep %d" % loop_extrusion_dynamics['reset_timestep'])
            #run_time = run_time - loop_extrusion_dynamics['reset_timestep']
        #except:
            #lmp.command("reset_timestep 0")

        # Start Loop extrusion dynamics
        if to_dump:
            lmp.command("undump 1")
            #lmp.command("dump    1       all    custom    %i   %sloop_extrusion_MD_*.XYZ  id  xu yu zu" % (to_dump,lammps_folder))
            lmp.command("dump    1       all    custom    %i   %sloop_extrusion_MD_*.lammpstrj  id  xu yu zu" % (to_dump,lammps_folder))
            lmp.command("dump_modify 1 format line \"%d %.5f %.5f %.5f\" sort id append no")
            
        # Get the positions of the fixed extruders and 
        extruders_positions = []
        extruders_lifetimes = []
        extruders_target_lifetimes = []
        right_bumps = []
        left_bumps = []
        right_bumps_moving = []
        left_bumps_moving = []                
        
        try:
            print("Defined barriers",loop_extrusion_dynamics['barriers'])
        except:
            pass
        
        try:
            print("Defined topology",loop_extrusion_dynamics['topology'])
        except:
            loop_extrusion_dynamics['topology'] = "Linear"
        
        # Randomly extract starting point of the extrusion dynamics between start and stop
        try:
            fixed_extruders = loop_extrusion_dynamics['fixed_extruders']
        except:
            fixed_extruders = []

        ### Define active barriers ###
        if "expression_rate" in loop_extrusion_dynamics:
            if not "expression_barriers" in loop_extrusion_dynamics:
                print("Define which barriers are expression related and which are not")
                exit(1)
            random_number = uniform(0, 1)                
            if loop_extrusion_dynamics["expression_rate"] > random_number:
                print("Neutralizing all expression_barriers for this simulations. Expression_rate = ", loop_extrusion_dynamics["expression_rate"])
                for iBarrier in range(len(loop_extrusion_dynamics['barriers'])):
                    if loop_extrusion_dynamics["expression_barriers"][iBarrier] == True:
                        loop_extrusion_dynamics['barriers_left_permeability'][iBarrier]  = 1
                        loop_extrusion_dynamics['barriers_right_permeability'][iBarrier] = 1
                        loop_extrusion_dynamics['lifetimeAtBarriersRight'][iBarrier]     = 0
                        loop_extrusion_dynamics['lifetimeAtBarriersLeft'][iBarrier]      = 0                        
            else:
                print("Keeping all expression_barriers for this simulations. Expression_rate = ", loop_extrusion_dynamics["expression_rate"])
                        
        print("### BEGIN Defined barriers ###")
        print("#Barrier left_permeability right_permeability lifetimeAtBarrierLeft lifetimeAtBarrierRight")
        if 'barriers' in loop_extrusion_dynamics:
            for barrier in loop_extrusion_dynamics['barriers']:
                left_permeability    = "NA"
                right_permeability   = "NA"
                barrierLifetimeLeft  = "NA"
                barrierLifetimeRight = "NA"
                expression_barrier   = "NA"
                if 'barriers_left_permeability' in loop_extrusion_dynamics:
                    left_permeability    = loop_extrusion_dynamics['barriers_left_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)]
                if 'barriers_right_permeability' in loop_extrusion_dynamics:
                    right_permeability   = loop_extrusion_dynamics['barriers_right_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)]
                if 'lifetimeAtBarriersRight' in loop_extrusion_dynamics:
                    barrierLifetimeRight = loop_extrusion_dynamics['lifetimeAtBarriersRight'][loop_extrusion_dynamics['barriers'].index(barrier)]                    
                if 'lifetimeAtBarriersLeft' in loop_extrusion_dynamics:
                    barrierLifetimeLeft  = loop_extrusion_dynamics['lifetimeAtBarriersLeft'][loop_extrusion_dynamics['barriers'].index(barrier)]
                if 'expression_barriers' in loop_extrusion_dynamics:
                    expression_barrier  = loop_extrusion_dynamics['expression_barriers'][loop_extrusion_dynamics['barriers'].index(barrier)]                    
                print(barrier, left_permeability, right_permeability, barrierLifetimeLeft, barrierLifetimeRight,"extrusionTimes","expression barrier:",expression_barrier)
        else:
            print("You didn't define any barriers for loop-extruders")
            loop_extrusion_dynamics['barriers_left_permeability']  = []
            loop_extrusion_dynamics['barriers_right_permeability'] = []
            loop_extrusion_dynamics['barriers']                    = []
        print("### END Defined barriers ###")

        """

        print("### BEGIN Defined barriers: ###")
        print("#Barrier left_permeability right_permeability")
        for barrier in loop_extrusion_dynamics['barriers']:
            print(barrier, loop_extrusion_dynamics['barriers_left_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)],loop_extrusion_dynamics['barriers_right_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)])
        print("### END Defined barriers: ###")
        """

        barriersOccupation = []
        extruderOnBarrierLeft   = []
        extruderOnBarrierRight  = []
        if 'barriers' in loop_extrusion_dynamics:
            for barrier in loop_extrusion_dynamics['barriers']:
                #if loop_extrusion_dynamics['barriers_left_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)] != 1.0 or loop_extrusion_dynamics['barriers_right_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)] != 1.0:
                #    barriersOccupation.append(barrier)
                extruderOnBarrierLeft.append(-1)
                extruderOnBarrierRight.append(-1)
            

        occupied_positions = list(chain(*fixed_extruders))+barriersOccupation
        print("Barriers' occupied positions",sorted(occupied_positions))
        lifetimeFactor = 1 #int(log(loop_extrusion_dynamics['lifetime'])/log(10))
        print("Lifetime ",loop_extrusion_dynamics['lifetime'])
        print("Lifetime factor ",lifetimeFactor)
                            
        sys.stdout.flush()        

        print("")
        print("### BEGIN Extruders loading sites ###")
        print("#Extruders_loading_sites")
        if 'ExtrudersLoadingSites' in loop_extrusion_dynamics:
            for site in loop_extrusion_dynamics['ExtrudersLoadingSites']:
                print(site)
        else:
            print("You didn't define any extruders loading sites")
        print("### END Defined Extruders loading sites ###")

        print("### BEGIN Positioning extruders ###")
        offset = 0 
        if 'ExtrudersLoadingSites' in loop_extrusion_dynamics:
            print("Positioning extruders on determined loading sites:")
            extrudersToLoad = []
            for c in range(len(loop_extrusion_dynamics['chrlength'])):
                nextruders    = int(loop_extrusion_dynamics['chrlength'][c]/loop_extrusion_dynamics['separation'])
                nLoadingSites = 0
                print("Number of extruders ",nextruders,"for copy",c+1)
                for site in loop_extrusion_dynamics['ExtrudersLoadingSites']:
                    if offset < site and site <= (offset + loop_extrusion_dynamics['chrlength'][c]):
                        
                        print(offset,site,offset + loop_extrusion_dynamics['chrlength'][c])
                        nLoadingSites += 1
                        new_positions    = [site,site+1]
                        extruders_positions.append(new_positions)
                        occupied_positions = occupied_positions + new_positions
                        print("Occupied positions",sorted(occupied_positions))

                        # Initialise the lifetime of each extruder and the flag to mark it they bumped on a barrier on the right or the left
                        extruders_lifetimes.append(int(0))
                        if lifetimeExponential == 1:
                            lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                            while lifetime == 0:
                                lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                        else:
                            lifetime = loop_extrusion_dynamics['lifetime']
                        extruders_target_lifetimes.append(lifetime)
                        right_bumps.append(int(0))
                        left_bumps.append(int(0))
                        right_bumps_moving.append(int(0))
                        left_bumps_moving.append(int(0))                        
                        if nLoadingSites == nextruders:
                            break
                offset += loop_extrusion_dynamics['chrlength'][c]
                extrudersToLoad.append(nextruders-nLoadingSites)
                print("Still ",extrudersToLoad[-1]," extruders to load over ",nextruders)
                
        else:
            print("Positioning extruders on random loading points:")
            for c in range(len(loop_extrusion_dynamics['chrlength'])): 
                nextruders = int(loop_extrusion_dynamics['chrlength'][c]/loop_extrusion_dynamics['separation'])
                print("Number of extruders ",nextruders,"for copy",c+1)       
                for extruder in range(nextruders):
                    print("Positioning extruder",extruder+1)

                    new_positions = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][c],distances)
                    new_positions[0] = offset + new_positions[0]
                    new_positions[1] = offset + new_positions[1]
                    while (new_positions[0] in occupied_positions) or (new_positions[1] in occupied_positions):
                        new_positions = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][0],distances)
                        if (new_positions[0] in fixed_extruders) or (new_positions[1] in fixed_extruders):
                            print("New positions",new_positions,"discarded")
                            sys.stdout.flush()
                    extruders_positions.append(new_positions)
                    occupied_positions = occupied_positions + new_positions
                    print("Occupied positions",sorted(occupied_positions))

                    # Initialise the lifetime of each extruder and the flag to mark it they bumped on a barrier on the right or the left
                    extruders_lifetimes.append(int(0))
                    if lifetimeExponential == 1:
                        lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                        while lifetime == 0:
                            lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                    else:
                        lifetime = loop_extrusion_dynamics['lifetime']
                    extruders_target_lifetimes.append(lifetime)
                    right_bumps.append(int(0))
                    left_bumps.append(int(0))
                    right_bumps_moving.append(int(0))
                    left_bumps_moving.append(int(0))                    
                offset += loop_extrusion_dynamics['chrlength'][c]
        print("### END Positioning extruders ###")
        extruders_positions = sorted(extruders_positions, key=itemgetter(0))
        print("Initial extruders' positions (All %d)"      % (len(extruders_positions)),extruders_positions)        
        print("Initial extruders' lifetimes (Variable %d)" % (len(extruders_lifetimes)),extruders_lifetimes)
        print("Initial extruders' target lifetimes (Variable %d)" % (len(extruders_target_lifetimes)),extruders_lifetimes)
        if 'ExtrudersLoadingSites' in loop_extrusion_dynamics:
            print("Extruders still to load per chain ",extrudersToLoad)
        #exit(1)
            
        if 'stalling_pairs'  in loop_extrusion_dynamics:
            stalling_pairs_lifetimes  = {}
            stalling_pairs_occupation = {}
            print("#Stalling_pair stalling_time current_stalling_time occupied_flag")
            for pair in loop_extrusion_dynamics['stalling_pairs']:
                stalling_pairs_lifetimes[pair] = 0
                stalling_pairs_occupation[pair] = -1
                print(pair, loop_extrusion_dynamics['stalling_pairs'][pair],stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
            print(stalling_pairs_lifetimes)
            print(stalling_pairs_occupation)

        print("Define the fixed extruders once for all")
        fixed_extruder_number=0
        for particle1,particle2 in fixed_extruders:
            fixed_extruder_number += 1
            print("# fix fixed_LE%i all restrain bond %i  %i %f %f %f" % (fixed_extruder_number,
                                                                          particle1,
                                                                          particle2,
                                                                          loop_extrusion_dynamics['attraction_strength'],
                                                                          loop_extrusion_dynamics['attraction_strength'],
                                                                          loop_extrusion_dynamics['equilibrium_distance']))
            
            lmp.command("fix fixed_LE%i all restrain bond %i  %i %f %f %f" % (fixed_extruder_number,
                                                                              particle1,
                                                                              particle2,
                                                                              loop_extrusion_dynamics['attraction_strength'],
                                                                              loop_extrusion_dynamics['attraction_strength'],
                                                                              loop_extrusion_dynamics['equilibrium_distance']))

            
        print("Defined",fixed_extruder_number,"fixed extruders")

        gene_PolII_occupation = {}
        moving_barriers       = []
        extruderOnBarrierLeftMoving  = []
        extruderOnBarrierRightMoving = []
        if "genes" in loop_extrusion_dynamics:
            for gene in loop_extrusion_dynamics["genes"]:
                gene_PolII_occupation[gene] = {}
                if "PolII_occupancy_per_gene" in loop_extrusion_dynamics:
                    if gene in loop_extrusion_dynamics["PolII_occupancy_per_gene"]:
                        PolII_occupancy = loop_extrusion_dynamics["PolII_occupancy_per_gene"][gene]
                    else:
                        PolII_occupancy = 0.1
                for particle in loop_extrusion_dynamics["genes"][gene]:                    
                    gene_PolII_occupation[gene][particle] = 0
                    randomProb = uniform(0,1)
                    if randomProb <= PolII_occupancy:
                        gene_PolII_occupation[gene][particle] = 1
                        moving_barriers.append(particle)
                        extruderOnBarrierLeftMoving.append(-1)
                        extruderOnBarrierRightMoving.append(-1)      
            print("Moving barriers",moving_barriers)


        lmp.command("compute xu all property/atom xu")
        lmp.command("compute yu all property/atom yu")
        lmp.command("compute zu all property/atom zu")
            
        print("Define the variable extruders")        
        for LES in range(int(run_time/loop_extrusion_dynamics['extrusion_time'])):
            print("### Extrusion round",LES,"###")
            thermo_style="thermo_style   custom   step temp epair emol"

            if "PolII_speed" in loop_extrusion_dynamics:
                if LES % loop_extrusion_dynamics["PolII_speed"] == 0:
                    print("Updating moving barriers positions at step",LES)
                    for i in range(len(moving_barriers)):
                        b = moving_barriers[i]
                        for gene in loop_extrusion_dynamics["genes"]:
                            if b in loop_extrusion_dynamics["genes"][gene]:
                                bgene = gene
                                if loop_extrusion_dynamics["genes"][gene][0] <= loop_extrusion_dynamics["genes"][gene][1]:
                                    gene_direction = 1
                                else:
                                    gene_direction = -1
                        # If Poll2 reaches the end of the gene, it is brought again at the beginning
                        if moving_barriers[i] + gene_direction == loop_extrusion_dynamics["genes"][gene][-1]:
                            moving_barriers[i] = loop_extrusion_dynamics["genes"][gene][0]
                        else:
                            moving_barriers[i] += gene_direction
                        extruderOnBarrierLeftMoving[i]  = -1
                        extruderOnBarrierRightMoving[i] = -1
                    print("Updated moving barriers",moving_barriers)
            
            # Update the bond restraint for variable extruders
            variable_extruder_number = 0
            for particle1,particle2 in extruders_positions:                    
                variable_extruder_number += 1
                print("# fix LE%i all restrain bond %i  %i %f %f %f %f" % (variable_extruder_number,
                                                                        particle1,
                                                                        particle2,
                                                                        loop_extrusion_dynamics['attraction_strength'],
                                                                        loop_extrusion_dynamics['attraction_strength'],
                                                                        1.0,
                                                                        loop_extrusion_dynamics['equilibrium_distance']))
                
                lmp.command("fix LE%i all restrain bond %i  %i %f %f %f %f" % (variable_extruder_number,
                                                                                  particle1,
                                                                                  particle2,
                                                                                  loop_extrusion_dynamics['attraction_strength'],
                                                                                  loop_extrusion_dynamics['attraction_strength'],
                                                                                  1.0,
                                                                                  loop_extrusion_dynamics['equilibrium_distance']))
                lmp.command("variable x%i equal c_xu[%i]" % (particle1, particle1))
                lmp.command("variable x%i equal c_xu[%i]" % (particle2, particle2))
                lmp.command("variable y%i equal c_yu[%i]" % (particle1, particle1))
                lmp.command("variable y%i equal c_yu[%i]" % (particle2, particle2))
                lmp.command("variable z%i equal c_zu[%i]" % (particle1, particle1))
                lmp.command("variable z%i equal c_zu[%i]" % (particle2, particle2))
                
                lmp.command("variable xLE%i equal v_x%i-v_x%i" % (variable_extruder_number, particle1, particle2))
                lmp.command("variable yLE%i equal v_y%i-v_y%i" % (variable_extruder_number, particle1, particle2))
                lmp.command("variable zLE%i equal v_z%i-v_z%i" % (variable_extruder_number, particle1, particle2))
                lmp.command("variable dist_%i_%i equal sqrt(v_xLE%i*v_xLE%i+v_yLE%i*v_yLE%i+v_zLE%i*v_zLE%i)" % (particle1,
                                                                                                                 particle2,
                                                                                                                 variable_extruder_number,
                                                                                                                 variable_extruder_number,
                                                                                                                 variable_extruder_number,
                                                                                                                 variable_extruder_number,
                                                                                                                 variable_extruder_number,
                                                                                                                 variable_extruder_number))
                thermo_style += " v_dist_%i_%i" % (particle1, particle2)
            print("Defined",variable_extruder_number,"variable extruders")

            lmp.command("%s" % thermo_style)
            # Doing the LES
            if '1D_run' in loop_extrusion_dynamics:
                lmp.command("run 0")
            else:
                lmp.command("run %i" % loop_extrusion_dynamics['extrusion_time'])

            #exit(1)
            #lmp.command("fix f_unwr all store/state 1 xu yu zu")
            xc_tmp = np.array(lmp.extract_fix("f_unwr",1,2).contents[0:(natoms-2)])
            distances = compute_particles_distance(xc_tmp)
            #print(distances)
                
            # update the lifetime of each extruder
            for extruder in range(len(extruders_positions)):
                extruders_lifetimes[extruder] = extruders_lifetimes[extruder] + 1
            
            # Remove the bond restraints of variable extruders!
            loop_number = 1
            for particle1,particle2 in extruders_positions:
                #print("# unfix LE%i" % (loop_number))
                lmp.command("unfix LE%i" % (loop_number))

                loop_number += 1

            # Update the particles involved in the loop extrusion interaction:
            # decrease the initial_start by one until you get to start
            # increase the initial_stop by one until you get to stop
            extruders_to_relocate = [1]*len(extruders_positions) # 0 if the extruder needs to be relocated and 1 if it doesn't!
            force_extruders_to_relocate = [1]*len(extruders_positions) # 0 if the extruder needs to be relocated and 1 if it doesn't!
            for extruder in range(len(extruders_positions)):
                print("")
                print("#Moving extruder",extruder)
                
                # Keep in memory the current positions
                tmp_extruder = extruders_positions[extruder].copy()
                # Keep in memory the chromosome limits
                total = 0
                start = 1
                nchr  = 0
                for c in loop_extrusion_dynamics['chrlength']:
                    nchr += 1
                    stop  = start + c - 1
                    #print("Chromosome",nchr,"goes from bead",start,"to bead",stop)
                    if start <= extruders_positions[extruder][0] and extruders_positions[extruder][0] <= stop:
                        break                    
                    start = stop + 1                    
                print("Chromosome",nchr,"from bead",start,"to bead",stop,"includes the extruder of position",extruders_positions[extruder])            
                
                # 1. Propose a move of the extruder with probabilities "left_extrusion_rate' or 'right_extrusion_rate'
                # and distinguishing in linear or ring topology
                if loop_extrusion_dynamics['topology'] == "Linear":
                    random_number = uniform(0, 1)
                    if right_bumps[extruder] == 1:
                        random_number = 0
                    if random_number <= float(loop_extrusion_dynamics["left_extrusion_rate"]):       
                        # If the left part reaches the start of the chromosome, put the extruder in another position and re-initialize its lifetime -> Routine to relocate extruder
                        if extruders_positions[extruder][0] > start:
                            extruders_positions[extruder][0] -= 1
                            #print("Propose moving the left arm of the extruder(",random_number,"<=",loop_extrusion_dynamics["left_extrusion_rate"],")",extruder,"from",tmp_extruder[0],"to",extruders_positions[extruder][0])
                        else:
                            #extruders_to_relocate[extruder] = 0 # 0 if the extruder needs to be relocated and 1 if it is doesn't!
                            force_extruders_to_relocate[extruder] = 0 # 0 if the extruder needs to be relocated and 1 if it is doesn't!
                            print("Relocate the extruder",extruder,"because it reached the start of the chain",tmp_extruder[0])
                            
                    random_number = uniform(0, 1)
                    if left_bumps[extruder] == 1:
                        random_number = 0
                    if random_number <= float(loop_extrusion_dynamics["right_extrusion_rate"]):
                        # If the right part reaches the end of the chromosome, put the extruder in another position and re-initialize its lifetime -> Routine to relocate extruder
                        if extruders_positions[extruder][1] < stop:
                            extruders_positions[extruder][1] += 1
                            #print("Propose moving the right arm of the extruder(",random_number,"<=",loop_extrusion_dynamics["right_extrusion_rate"],")",extruder,"from",tmp_extruder[1],"to",extruders_positions[extruder][1])
                        else:
                            #extruders_to_relocate[extruder] = 0 # 0 if the extruder needs to be relocated and 1 if it is doesn't!
                            force_extruders_to_relocate[extruder] = 0 # 0 if the extruder needs to be relocated and 1 if it is doesn't!
                            print("Relocate the extruder",extruder,"because it reached the end of the chain",tmp_extruder[1])                            

                # Move the extruder if it doesn't hit the chromosome limits
                if loop_extrusion_dynamics['topology'] == "Ring":
                    # If the left part reaches the start of the chromosome -> Go to the end
                    random_number = uniform(0, 1)
                    if random_number <= loop_extrusion_dynamics['left_extrusion_rate']:
                        if tmp_extruder[0] > start:
                            extruders_positions[extruder][0] -= 1
                        elif tmp_extruder[0] == start:
                            extruders_positions[extruder][0] = stop
                        elif extruders_positions[extruder][0] == tmp_extruder[1]:
                            extruders_positions[extruder][0] = tmp_extruder[0]
                            
                    # If the right part reaches the end of the chromosome -> Go to start
                    random_number = uniform(0, 1)
                    if random_number <= loop_extrusion_dynamics['right_extrusion_rate']:
                        if tmp_extruder[1] <  stop:
                            extruders_positions[extruder][1] += 1
                        if tmp_extruder[1] == stop:
                            extruders_positions[extruder][1] = start
                        if extruders_positions[extruder][1] == tmp_extruder[0]:
                            extruders_positions[extruder][1] = tmp_extruder[1]
                    #print("Propose moving the extruder",extruder,"from",tmp_extruder,"to",extruders_positions[extruder])
                            
                # 2. If the extruder bumps into another extruder (fixed or variable)
                tmp_extruders_positions = [extruders_positions[x] for x in range(len(extruders_positions)) if x != extruder]

                occupied_positions = list(chain(*fixed_extruders))+list(chain(*tmp_extruders_positions))
                #print("Proposed extruder positions",extruders_positions[extruder])
                #print("Occupied_positions (Extruder excluded)",occupied_positions)

                if   loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "stalling":
                    #Option 1 bring it back
                    if extruders_positions[extruder][0] in occupied_positions:
                        print("The left  part of the extruder bumped into an occupied position: the new position",extruders_positions[extruder][0],"is brought to the previous one",tmp_extruder[0])
                        extruders_positions[extruder][0] = tmp_extruder[0]
                    if extruders_positions[extruder][1] in occupied_positions:
                        print("The right part of the extruder bumped into an occupied position: the new position",extruders_positions[extruder][1],"is brought to the previous one",tmp_extruder[1])
                        extruders_positions[extruder][1] = tmp_extruder[1]
                        
                elif loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "relocating":
                    #Option 2 put it in another position and re-initialize its lifetime -> Routine to relocate extruder
                    if extruders_positions[extruder][0] in occupied_positions:
                        print("The left arm of the extruder",extruder,"bumped into an occupied position: the new position",extruders_positions[extruder][0],"it is relocated to a new random position")
                        extruders_to_relocate[extruder] = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                    if extruders_positions[extruder][1] in occupied_positions:
                        print("The right arm of the extruder",extruder,"bumped into an occupied position: the new position",extruders_positions[extruder][1],"it is relocated to a new random position")
                        extruders_to_relocate[extruder] = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!

                elif loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "crossing":
                    #Option 3 extruders can cross each other
                    if extruders_positions[extruder][0] in occupied_positions:
                        print("The left  arm of the extruder",extruder,"bumped into an occupied position, but they can cross each other")
                    if extruders_positions[extruder][1] in occupied_positions:
                        print("The right arm of the extruder",extruder,"bumped into an occupied position, but they can cross each other")

                # 3.5 if the extruder reaches a PolII moving barrier, the extruder will stop if it is in the direction of the transcription
                if extruders_positions[extruder][0] in moving_barriers:                    
                    perm = 0.2
                    # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability                    
                    # If the barrier is on the left of the extruders, which is extruding contrary to the chain index, we have to re-put the extruder forwards
                    print("Found a moving barrier coming from the right at monomer %d with permeability %f" % (extruders_positions[extruder][0],perm))
                    # If the extruder was already stopped right_bumps[extruder] == 1, we have to block it
                    randomProb = uniform(0,1)
                    if right_bumps_moving[extruder] == 1:
                        print("The extruder was already stopped at this barrier, it will stay here until the end of its lifetime ",extruders_target_lifetimes[extruder])
                        randomProb = 1.
                    if randomProb > perm:
                        # Option 1: The extruder stops at the barrier
                        right_bumps_moving[extruder] = 1
                        extruders_to_relocate[extruder]      = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                        # Now, if the extruders can cross each other, they tend to concentrate at barriers, to avoid that we forbid an extruder to stall at an occupied barrier and relocate it
                        if loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "crossing":
                            print("ExtrudeOnbarrier",extruderOnBarrierRightMoving)
                            print("Barriers",moving_barriers)
                            print("Index of the barrier",moving_barriers.index(extruders_positions[extruder][0]))
                            if extruderOnBarrierRightMoving[moving_barriers.index(extruders_positions[extruder][0])] != extruder and extruderOnBarrierRightMoving[moving_barriers.index(extruders_positions[extruder][0])] != -1:
                                print("When the extruders can cross each other, they tend to concentrate at barriers.")
                                print("To avoid that we forbid an extruder to stall at an occupied barrier and relocate it.")
                                print("Accordingly I will relocate extruder",extruder)
                                extruders_to_relocate[extruder]      = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                            else:
                                extruderOnBarrierRightMoving[moving_barriers.index(extruders_positions[extruder][0])] = extruder
                        print("The left  part of the extruder",extruder,"bumped in a moving barrier: the new position",extruders_positions[extruder][0],"is brought to the previous one",tmp_extruder[0])
                        extruders_positions[extruder][0] = tmp_extruder[0]
                    #exit(1)
                        
                if extruders_positions[extruder][1] in moving_barriers:
                    perm = 0.2
                    # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability
                    # If the barrier is on the right of the extruders, which is extruding with the chain index, we have to re-put the extruder backwards
                    print("Found a barrier coming from the left at monomer %d with permeability %f" % (extruders_positions[extruder][1],perm))
                    # If the extruder was already stopped left_bumps[extruder] == 1, we have to block it
                    randomProb = uniform(0,1)
                    if left_bumps_moving[extruder] == 1:
                        print("The extruder was already stopped at this barrier, it will stay here until the end of its lifetime ",extruders_target_lifetimes[extruder])
                        randomProb = 1
                    if randomProb > perm:
                        # Option 1: The extruder stops at the barrier
                        left_bumps_moving[extruder] = 1
                        # Avoid to relocate the extruder!
                        extruders_to_relocate[extruder] = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                        # Now, if the extruders can cross each other, they tend to concentrate at barriers, to avoid that we forbid an extruder to stall at an occupied barrier and relocate it
                        if loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "crossing":
                            print("ExtrudeOnbarrier",extruderOnBarrierLeftMoving)
                            print("Barriers",moving_barriers)
                            print("Index of the barrier",moving_barriers.index(extruders_positions[extruder][1]))
                            if extruderOnBarrierLeftMoving[moving_barriers.index(extruders_positions[extruder][1])] != extruder and extruderOnBarrierLeftMoving[moving_barriers.index(extruders_positions[extruder][1])] != -1:
                                print("When the extruders can cross each other, they tend to concentrate at barriers.")
                                print("To avoid that we forbid an extruder to stall at an occupied barrier and relocate it.")
                                print("Accordingly I will relocate extruder",extruder)
                                extruders_to_relocate[extruder]      = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                            else:
                                extruderOnBarrierLeftMoving[moving_barriers.index(extruders_positions[extruder][1])] = extruder
                        print("The right part of the extruder",extruder,"bumped in a barrier: the new position",extruders_positions[extruder][1],"is brought to the previous one",tmp_extruder[1])
                        extruders_positions[extruder][1] = tmp_extruder[1]

                        
                # 3. if the extruder reaches a barrier it is stop or not depending on the permeability of the barrier
                if   loop_extrusion_dynamics["loop_extruder_barrier_encounter_rule"] == "stalling":

                    if extruders_positions[extruder][0] in loop_extrusion_dynamics['barriers']:
                        perm = loop_extrusion_dynamics['barriers_right_permeability'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])]
                        if 'lifetimeAtBarriersRight' in loop_extrusion_dynamics:
                            barrierLifetimeRight = loop_extrusion_dynamics['lifetimeAtBarriersRight'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])]
                        # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability
                        # If the barrier is on the left of the extruders, which is extruding contrary to the chain index, we have to re-put the extruder forwards
                        print("Found a barrier coming from the right at monomer %d with permeability %f" % (extruders_positions[extruder][0],perm))
                        # If the extruder was already stopped right_bumps[extruder] == 1, we have to block it
                        randomProb = uniform(0,1)
                        if right_bumps[extruder] == 1:
                            print("The extruder was already stopped at this barrier, it will stay here until the end of its lifetime ",extruders_target_lifetimes[extruder])
                            if 'stalling' in loop_extrusion_dynamics:
                                extruders_positions[extruder][0] = tmp_extruder[0]
                                extruders_positions[extruder][1] = tmp_extruder[1]
                            randomProb = 1.
                        if randomProb > perm:
                            # Option 1: The extruder stops at the barrier
                            if right_bumps[extruder] != 1:
                                tmpLifetime = extruders_lifetimes[extruder] 
                                extruders_target_lifetimes[extruder] = tmpLifetime + barrierLifetimeRight
                            right_bumps[extruder] = 1
                            extruders_to_relocate[extruder]      = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                            # Now, if the extruders can cross each other, they tend to concentrate at barriers, to avoid that we forbid an extruder to stall at an occupied barrier and relocate it
                            if loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "crossing":
                                print("ExtrudeOnbarrier",extruderOnBarrierRight)
                                print("Barriers",loop_extrusion_dynamics['barriers'])
                                print("Index of the barrier",loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0]))
                                if extruderOnBarrierRight[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])] != extruder and extruderOnBarrierRight[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])] != -1:
                                    print("When the extruders can cross each other, they tend to concentrate at barriers.")
                                    print("To avoid that we forbid an extruder to stall at an occupied barrier and relocate it.")
                                    print("Accordingly I will relocate extruder",extruder)
                                    extruders_to_relocate[extruder]      = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                                else:
                                    extruderOnBarrierRight[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])] = extruder
                            print("The left  part of the extruder",extruder,"bumped in a barrier: the new position",extruders_positions[extruder][0],"is brought to the previous one",tmp_extruder[0])
                            extruders_positions[extruder][0] = tmp_extruder[0]

                    if extruders_positions[extruder][1] in loop_extrusion_dynamics['barriers']:
                        perm = loop_extrusion_dynamics['barriers_left_permeability'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])]
                        if 'lifetimeAtBarriersLeft' in loop_extrusion_dynamics:
                            barrierLifetimeLeft = loop_extrusion_dynamics['lifetimeAtBarriersLeft'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])]
                        # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability
                        # If the barrier is on the right of the extruders, which is extruding with the chain index, we have to re-put the extruder backwards
                        print("Found a barrier coming from the left at monomer %d with permeability %f" % (extruders_positions[extruder][1],perm))
                        # If the extruder was already stopped left_bumps[extruder] == 1, we have to block it
                        randomProb = uniform(0,1)
                        if left_bumps[extruder] == 1:
                            print("The extruder was already stopped at this barrier, it will stay here until the end of its lifetime ",extruders_target_lifetimes[extruder])
                            if 'stalling' in loop_extrusion_dynamics:
                                extruders_positions[extruder][0] = tmp_extruder[0]
                                extruders_positions[extruder][1] = tmp_extruder[1]
                            randomProb = 1
                        if randomProb > perm:
                            # Option 1: The extruder stops at the barrier
                            if left_bumps[extruder] != 1:
                                tmpLifetime = extruders_lifetimes[extruder] 
                                extruders_target_lifetimes[extruder] = tmpLifetime + barrierLifetimeLeft
                            left_bumps[extruder] = 1
                            # Avoid to relocate the extruder!
                            extruders_to_relocate[extruder] = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                            # Now, if the extruders can cross each other, they tend to concentrate at barriers, to avoid that we forbid an extruder to stall at an occupied barrier and relocate it
                            if loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "crossing":
                                print("ExtrudeOnbarrier",extruderOnBarrierLeft)
                                print("Barriers",loop_extrusion_dynamics['barriers'])
                                print("Index of the barrier",loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1]))
                                if extruderOnBarrierLeft[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])] != extruder and extruderOnBarrierLeft[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])] != -1:
                                    print("When the extruders can cross each other, they tend to concentrate at barriers.")
                                    print("To avoid that we forbid an extruder to stall at an occupied barrier and relocate it.")
                                    print("Accordingly I will relocate extruder",extruder)
                                    extruders_to_relocate[extruder]      = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                                else:
                                    extruderOnBarrierLeft[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])] = extruder
                            print("The right part of the extruder",extruder,"bumped in a barrier: the new position",extruders_positions[extruder][1],"is brought to the previous one",tmp_extruder[1])
                            extruders_positions[extruder][1] = tmp_extruder[1]


                elif loop_extrusion_dynamics["loop_extruder_barrier_encounter_rule"] == "relocating":
                    if extruders_positions[extruder][0] in loop_extrusion_dynamics['barriers']:
                        perm = loop_extrusion_dynamics['barriers_right_permeability'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])]
                        # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability
                        # If the barrier is on the left of the extruders, which is extruding contrary to the chain index, we have to re-put the extruder forwards
                        print("Found a barrier coming from the right at monomer %d with permeability %f" % (extruders_positions[extruder][0],perm))
                        if uniform(0,1) > perm:
                            print("The left  part of the extruder",extruder,"bumped in a barrier and it will be relocated")
                            extruders_to_relocate[extruder] = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!

                    if extruders_positions[extruder][1] in loop_extrusion_dynamics['barriers']:
                        perm = loop_extrusion_dynamics['barriers_left_permeability'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])]
                        # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability
                        # If the barrier is on the right of the extruders, which is extruding with the chain index, we have to re-put the extruder backwards
                        print("Found a barrier coming from the left at monomer %d with permeability %f" % (extruders_positions[extruder][1],perm))
                        if uniform(0,1) > perm:
                            print("The right part of the extruder",extruder,"bumped in a barrier and it will be relocated")
                            extruders_to_relocate[extruder] = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!                            

                # 3b. If the extruder connects two stalling_pairs it will stay there for the life of the stalling pair
                if 'stalling_pairs'  in loop_extrusion_dynamics:
                    for pair in loop_extrusion_dynamics['stalling_pairs']:
                        if (tmp_extruder[0] == pair[0] and tmp_extruder[1] == pair[1]) or (tmp_extruder[0] == pair[1] and tmp_extruder[1] == pair[0]):
                            print("Extruder",extruder,"found a potential stalling pair",pair,loop_extrusion_dynamics['stalling_pairs'][pair])
                            # If the stalling pair is already occupied, just leave
                            if stalling_pairs_occupation[pair] != -1 and stalling_pairs_occupation[pair] != extruder:
                                print("But extruder",extruder,"found an occupied stalling pair",pair,stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
                                print("Accordingly, it will continue extruding")
                            else:
                                if stalling_pairs_lifetimes[pair] == loop_extrusion_dynamics['stalling_pairs'][pair]:
                                    print("Extruder",extruder,"stalled in a pair for the target lifetime")
                                    print(pair, loop_extrusion_dynamics['stalling_pairs'][pair],stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
                                    print("It will keep extruding now!")
                                    stalling_pairs_lifetimes[pair]  = 0
                                    stalling_pairs_occupation[pair] = -1
                                    print("The lifetime and occupation of the stalling pairs are re-initialized")
                                    print(pair,"lifetime",stalling_pairs_lifetimes[pair],"target-lifetime",loop_extrusion_dynamics['stalling_pairs'][pair],"occupation",stalling_pairs_occupation[pair])
                                else:
                                    print("Extruder",extruder,"stalled in a pair")
                                    print("#Stalling_pair stalling_time current_stalling_time occupied_flag")
                                    print(pair, loop_extrusion_dynamics['stalling_pairs'][pair],stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
                                    sys.stdout.flush()
                                    stalling_pairs_occupation[pair] = extruder
                                    print("Pause the extruder in this position")
                                    extruders_positions[extruder][0] = tmp_extruder[0]
                                    extruders_positions[extruder][1] = tmp_extruder[1]
                                    stalling_pairs_lifetimes[pair] = stalling_pairs_lifetimes[pair] + 1
                                    print("Avoid to relocate the extruder")
                                    extruders_to_relocate[extruder] = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!

                        """
                        if (extruders_positions[extruder][0] == pair[0] and extruders_positions[extruder][1] == pair[1]) or (extruders_positions[extruder][0] == pair[1] and extruders_positions[extruder][1] == pair[0]):
                            if stalling_pairs_occupation[pair] != -1 and stalling_pairs_occupation[pair] != extruder:
                                continue
                            else:
                                if stalling_pairs_lifetimes[pair] == loop_extrusion_dynamics['stalling_pairs'][pair]:
                                    print("Extruder",extruder,"stalled in a pair for the target lifetime")
                                    print(pair, loop_extrusion_dynamics['stalling_pairs'][pair],stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
                                    print("It will keep extruding now!")
                                    stalling_pairs_lifetimes[pair] = 0
                                else:
                                    print("Extruder",extruder,"stalled in a pair")
                                    print("#Stalling_pair stalling_time current_stalling_time occupied_flag")
                                    print(pair, loop_extrusion_dynamics['stalling_pairs'][pair],stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
                                    sys.stdout.flush()
                                    stalling_pairs_occupation[pair] = extruder
                                    stalling_pairs_lifetimes[pair]   = stalling_pairs_lifetimes[pair] + 1 
                                    print("Pause the extruder in this position")
                                    extruders_positions[extruder][0] = tmp_extruder[0]
                                    extruders_positions[extruder][1] = tmp_extruder[1]
                                    stalling_pairs_lifetimes[pair] = stalling_pairs_lifetimes[pair] + 1
                                    print("Avoid to relocate the extruder")
                                    extruders_to_relocate[extruder] = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!                              
                        """
                # 3c. If the extruder connects two switchOn_pairs it will apply an attractive interaction between them as in the compartmentalization case,
                # So the structure of loop_extrusion_dynamics['switchOn_pairs'] is the same of compartmentalization['interactions']
                if 'switchOn_pairs'  in loop_extrusion_dynamics:
                    for pair in loop_extrusion_dynamics['switchOn_pairs']:
                        #{(1196,1198) : ["attraction",4.0000000000]}
                        t1      = pair[0]
                        t2      = pair[1]
                        epsilon = loop_extrusion_dynamics['switchOn_pairs'][pair][1]
                        sigma = 1.0
                        rc = sigma * 2.5
                        lmp.command("pair_coeff %s %s lj/cut %s %s %s" % (t1,t2,epsilon,sigma,rc))

                # XXX. If the new bond is larger than 100nm
                #if loop_extrusion_dynamics['max_distance_to_create']:

                # 4. If the extruder reached its lifetime, put it in another position and re-initialize its lifetime -> Routine to relocate extruder
                if extruders_lifetimes[extruder] >= extruders_target_lifetimes[extruder]:
                    extruders_to_relocate[extruder] = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!

                # Routine to relocate extruder
                if extruders_to_relocate[extruder] == 0 or force_extruders_to_relocate[extruder] == 0:  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                    print("Relocating extruder ",extruder," lifetime ",extruders_lifetimes[extruder]," Target-lifetime ",extruders_target_lifetimes[extruder])                    
                    tmp_extruders_positions = [extruders_positions[x] for x in range(len(extruders_positions)) if x != extruder]
                    occupied_positions = list(chain(*fixed_extruders))+list(chain(*tmp_extruders_positions))+barriersOccupation
                    print("Occupied_positions (Extruder excluded)",sorted(occupied_positions))

                    if 'ExtrudersLoadingSites' in loop_extrusion_dynamics:
                        offset = 0
                        for c in range(len(loop_extrusion_dynamics['chrlength'])):                                
                            if c != (nchr-1):
                                continue
                            nLoadingSites = 0
                            sites = []
                            print(sites)
                            for site in loop_extrusion_dynamics['ExtrudersLoadingSites']:
                                if offset < site and site <= (offset + loop_extrusion_dynamics['chrlength'][c]):
                                    sites.append(site)
                            sites = sample(sites, len(sites))
                            print(sites)

                            for site in sites:
                                print(offset,site,offset + loop_extrusion_dynamics['chrlength'][c])
                                new_positions    = [site,site+1]                                
                                extruders_positions[extruder] = new_positions
                                if (extruders_positions[extruder][0] in occupied_positions) or (extruders_positions[extruder][1] in occupied_positions):
                                    print("One of the proposed random positions",extruders_positions[extruder],"is occupied")
                                    continue
                                nLoadingSites += 1
                                break
                            if nLoadingSites == 0:
                                print("NOTE: not enough loading determined loading sites to load this extruders: Using a random one!")                                
                                extruders_positions[extruder]    = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][nchr-1],distances)
                                extruders_positions[extruder][0] = extruders_positions[extruder][0] + start
                                extruders_positions[extruder][1] = extruders_positions[extruder][1] + start
                                print("Proposed random position",extruders_positions[extruder])
                    
                                while (extruders_positions[extruder][0] in occupied_positions) or (extruders_positions[extruder][1] in occupied_positions):
                                    print("One of the proposed random positions",extruders_positions[extruder],"is occupied")
                                    extruders_positions[extruder]    = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][nchr-1],distances)
                                    extruders_positions[extruder][0] = extruders_positions[extruder][0] + start
                                    extruders_positions[extruder][1] = extruders_positions[extruder][1] + start
                                    print("Proposed random position",extruders_positions[extruder])
                                    print("")
                            offset += loop_extrusion_dynamics['chrlength'][c]
                    else:
                        extruders_positions[extruder]    = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][nchr-1],distances)
                        extruders_positions[extruder][0] = extruders_positions[extruder][0] + start
                        extruders_positions[extruder][1] = extruders_positions[extruder][1] + start
                        print("Proposed random position",extruders_positions[extruder])
                    
                        #and (start <= extruders_positions[extruder][1] and extruders_positions[extruder][1] <= stop)
                        #and (start <= extruders_positions[extruder][0] and extruders_positions[extruder][0] <= stop):
                        while (extruders_positions[extruder][0] in occupied_positions) or (extruders_positions[extruder][1] in occupied_positions):
                            print("One of the proposed random positions",extruders_positions[extruder],"is occupied")
                            extruders_positions[extruder]    = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][nchr-1],distances)
                            extruders_positions[extruder][0] = extruders_positions[extruder][0] + start
                            extruders_positions[extruder][1] = extruders_positions[extruder][1] + start
                            print("Proposed random position",extruders_positions[extruder])

                    # Re-initialise the lifetime of the extruder
                    extruders_lifetimes[extruder] = 0
                    right_bumps[extruder] = 0
                    left_bumps[extruder] = 0
                    right_bumps_moving[extruder] = 0
                    left_bumps_moving[extruder] = 0                    
                    if lifetimeExponential == 1:
                        lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                        while lifetime == 0:
                            lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                    else:
                        lifetime = loop_extrusion_dynamics['lifetime']
                    extruders_target_lifetimes[extruder] = lifetime

                    # Re-initialise the occupation of barriers on the left, if the extruder was stalling there
                    for i in range(len(extruderOnBarrierLeft)):
                        if extruderOnBarrierLeft[i] == extruder:                            
                            extruderOnBarrierLeft[i] = -1
                            print("Barrier",loop_extrusion_dynamics['barriers'][i],"is now free of extruders at the left!")
                    # Re-initialise the occupation of barriers on the right, if the extruder was stalling there
                    for i in range(len(extruderOnBarrierRight)):
                        if extruderOnBarrierRight[i] == extruder:                            
                            extruderOnBarrierRight[i] = -1
                            print("Barrier",loop_extrusion_dynamics['barriers'][i],"is now free of extruders at the right!")
                    # Re-initialise the occupation of barriers on the left, if the extruder was stalling there
                    for i in range(len(extruderOnBarrierLeftMoving)):
                        if extruderOnBarrierLeftMoving[i] == extruder:                            
                            extruderOnBarrierLeftMoving[i] = -1
                            print("Barrier",moving_barriers[i],"is now free of extruders at the left!")
                    # Re-initialise the occupation of barriers on the right, if the extruder was stalling there
                    for i in range(len(extruderOnBarrierRightMoving)):
                        if extruderOnBarrierRightMoving[i] == extruder:                            
                            extruderOnBarrierRightMoving[i] = -1
                            print("Barrier",loop_extrusion_dynamics['barriers'][i],"is now free of extruders at the right!")

                    #extruders_positions = sorted(extruders_positions, key=itemgetter(0))
                    #print("Extruder",extruder,"relocated from",tmp_extruder,"to",extruders_positions[extruder])

            print("Extruders positions at step",LES,extruders_positions)
            print("Extruders lifetimes at step",LES,extruders_lifetimes)
            print("Extruders target lifetimes at step",LES,extruders_target_lifetimes)
            print("Extruder on barriers left", LES,extruderOnBarrierLeft)
            print("Extruder on barriers right",LES,extruderOnBarrierRight)
            print("Left bumps at step",LES,left_bumps)
            print("Right bumps at step",LES,right_bumps)
            print("Left bumps moving barriers at step",LES,left_bumps_moving)            
            print("Right bumps moving barriers at step",LES,right_bumps_moving)
            sys.stdout.flush()

            if 'ExtrudersLoadingSites' in loop_extrusion_dynamics and sum(extrudersToLoad) != 0:
                print("Try to position the missing extruders at determined loading sites")
                offset = 0
                occupied_positions = list(chain(*fixed_extruders))+list(chain(*extruders_positions))+barriersOccupation
                extrudersToLoadTmp = []
                for c in range(len(loop_extrusion_dynamics['chrlength'])):        
                    nextruders    = extrudersToLoad[c]
                    if nextruders == 0:
                        extrudersToLoadTmp.append(0)
                        continue
                    nLoadingSites = 0
                    print("Number of extruders ",nextruders,"for copy",c+1)
                    sites = []
                    print(sites)
                    for site in loop_extrusion_dynamics['ExtrudersLoadingSites']:
                        if offset < site and site <= (offset + loop_extrusion_dynamics['chrlength'][c]):
                            sites.append(site)
                    sites = sample(sites, len(sites))
                    print(sites)
                    for site in sites:
                        print(offset,site,offset + loop_extrusion_dynamics['chrlength'][c])
                        nLoadingSites += 1
                        new_positions    = [site,site+1]
                        extruders_positions.append(new_positions)
                        occupied_positions = occupied_positions + new_positions
                        print("Occupied positions",sorted(occupied_positions))
                        
                        # Initialise the lifetime of each extruder and the flag to mark it they bumped on a barrier on the right or the left
                        extruders_lifetimes.append(int(0))
                        if lifetimeExponential == 1:
                            lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                            while lifetime == 0:
                                lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                        else:
                            lifetime = loop_extrusion_dynamics['lifetime']
                        extruders_target_lifetimes.append(lifetime)
                        right_bumps.append(int(0))
                        left_bumps.append(int(0))
                        right_bumps_moving.append(int(0))
                        left_bumps_moving.append(int(0))                        
                        if nLoadingSites == nextruders:
                            break
                    offset += loop_extrusion_dynamics['chrlength'][c]
                    extrudersToLoadTmp.append(nextruders-nLoadingSites)
                    print("Still ",extrudersToLoadTmp[-1]," extruders to load over ",nextruders)                    
                    print("")
                extrudersToLoad = extrudersToLoadTmp.copy()
                print("Extruders still to load per chain ",extrudersToLoad)
            sys.stdout.flush()


    # Setup the pairs to co-localize using the COLVARS plug-in
    if loop_extrusion_dynamics:        
        
        np.random.seed(424242)
        
        print("left_extrusion_rate",loop_extrusion_dynamics["left_extrusion_rate"])
        print("right_extrusion_rate",loop_extrusion_dynamics["right_extrusion_rate"])
        
        if 'lifetimeConstant' in loop_extrusion_dynamics:
            lifetimeExponential = 0
            print("Target extruders lifetimes will be always equal to",loop_extrusion_dynamics['lifetime'])
        else:
            lifetimeExponential = 1
            print("Target extruders lifetimes will be drawn from an exponential distribution with average equal to",loop_extrusion_dynamics['lifetime'])

        natoms = int(lmp.get_natoms())
        print(natoms)
        lmp.command("fix f_unwr all store/state 1 xu yu zu")
        xc_tmp = np.array(lmp.extract_fix("f_unwr",1,2).contents[0:(natoms-2)])
        distances = compute_particles_distance(xc_tmp)
        #print(xc_tmp)
        #print(distances)
        
        #lmp.command("compute forces all pair/local fx fy fz")
        #lmp.command("dump forces all local 1000 pairwise_forces.dump index c_forces[1] c_forces[2] c_forces[3]")

        # Start relaxation step
        try:
            #lmp.command("reset_timestep 0")
            lmp.command("run %i" % loop_extrusion_dynamics['timesteps_relaxation'])
        except:
            pass

        #try :
            #lmp.command("reset_timestep %d" % loop_extrusion_dynamics['reset_timestep'])
            #run_time = run_time - loop_extrusion_dynamics['reset_timestep']
        #except:
            #lmp.command("reset_timestep 0")

        # Start Loop extrusion dynamics
        if to_dump:
            lmp.command("undump 1")
            #lmp.command("dump    1       all    custom    %i   %sloop_extrusion_MD_*.XYZ  id  xu yu zu" % (to_dump,lammps_folder))
            lmp.command("dump    1       all    custom    %i   %sloop_extrusion_MD_*.lammpstrj  id  xu yu zu" % (to_dump,lammps_folder))
            lmp.command("dump_modify 1 format line \"%d %.5f %.5f %.5f\" sort id append no")
            
        # Get the positions of the fixed extruders and 
        extruders_positions = []
        extruders_lifetimes = []
        extruders_target_lifetimes = []
        right_bumps = []
        left_bumps = []
        right_bumps_moving = []
        left_bumps_moving = []                
        
        try:
            print("Defined barriers",loop_extrusion_dynamics['barriers'])
        except:
            pass
        
        try:
            print("Defined topology",loop_extrusion_dynamics['topology'])
        except:
            loop_extrusion_dynamics['topology'] = "Linear"
        
        # Randomly extract starting point of the extrusion dynamics between start and stop
        try:
            fixed_extruders = loop_extrusion_dynamics['fixed_extruders']
        except:
            fixed_extruders = []

        ### Define active barriers ###
        if "expression_rate" in loop_extrusion_dynamics:
            if not "expression_barriers" in loop_extrusion_dynamics:
                print("Define which barriers are expression related and which are not")
                exit(1)
            random_number = uniform(0, 1)                
            if loop_extrusion_dynamics["expression_rate"] > random_number:
                print("Neutralizing all expression_barriers for this simulations. Expression_rate = ", loop_extrusion_dynamics["expression_rate"])
                for iBarrier in range(len(loop_extrusion_dynamics['barriers'])):
                    if loop_extrusion_dynamics["expression_barriers"][iBarrier] == True:
                        loop_extrusion_dynamics['barriers_left_permeability'][iBarrier]  = 1
                        loop_extrusion_dynamics['barriers_right_permeability'][iBarrier] = 1
                        loop_extrusion_dynamics['lifetimeAtBarriersRight'][iBarrier]     = 0
                        loop_extrusion_dynamics['lifetimeAtBarriersLeft'][iBarrier]      = 0                        
            else:
                print("Keeping all expression_barriers for this simulations. Expression_rate = ", loop_extrusion_dynamics["expression_rate"])
                        
        print("### BEGIN Defined barriers ###")
        print("#Barrier left_permeability right_permeability lifetimeAtBarrierLeft lifetimeAtBarrierRight")
        if 'barriers' in loop_extrusion_dynamics:
            for barrier in loop_extrusion_dynamics['barriers']:
                left_permeability    = "NA"
                right_permeability   = "NA"
                barrierLifetimeLeft  = "NA"
                barrierLifetimeRight = "NA"
                expression_barrier   = "NA"
                if 'barriers_left_permeability' in loop_extrusion_dynamics:
                    left_permeability    = loop_extrusion_dynamics['barriers_left_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)]
                if 'barriers_right_permeability' in loop_extrusion_dynamics:
                    right_permeability   = loop_extrusion_dynamics['barriers_right_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)]
                if 'lifetimeAtBarriersRight' in loop_extrusion_dynamics:
                    barrierLifetimeRight = loop_extrusion_dynamics['lifetimeAtBarriersRight'][loop_extrusion_dynamics['barriers'].index(barrier)]                    
                if 'lifetimeAtBarriersLeft' in loop_extrusion_dynamics:
                    barrierLifetimeLeft  = loop_extrusion_dynamics['lifetimeAtBarriersLeft'][loop_extrusion_dynamics['barriers'].index(barrier)]
                if 'expression_barriers' in loop_extrusion_dynamics:
                    expression_barrier  = loop_extrusion_dynamics['expression_barriers'][loop_extrusion_dynamics['barriers'].index(barrier)]                    
                print(barrier, left_permeability, right_permeability, barrierLifetimeLeft, barrierLifetimeRight,"extrusionTimes","expression barrier:",expression_barrier)
        else:
            print("You didn't define any barriers for loop-extruders")
            loop_extrusion_dynamics['barriers_left_permeability']  = []
            loop_extrusion_dynamics['barriers_right_permeability'] = []
            loop_extrusion_dynamics['barriers']                    = []
        print("### END Defined barriers ###")

        """

        print("### BEGIN Defined barriers: ###")
        print("#Barrier left_permeability right_permeability")
        for barrier in loop_extrusion_dynamics['barriers']:
            print(barrier, loop_extrusion_dynamics['barriers_left_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)],loop_extrusion_dynamics['barriers_right_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)])
        print("### END Defined barriers: ###")
        """

        barriersOccupation = []
        extruderOnBarrierLeft   = []
        extruderOnBarrierRight  = []
        if 'barriers' in loop_extrusion_dynamics:
            for barrier in loop_extrusion_dynamics['barriers']:
                #if loop_extrusion_dynamics['barriers_left_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)] != 1.0 or loop_extrusion_dynamics['barriers_right_permeability'][loop_extrusion_dynamics['barriers'].index(barrier)] != 1.0:
                #    barriersOccupation.append(barrier)
                extruderOnBarrierLeft.append(-1)
                extruderOnBarrierRight.append(-1)
            

        occupied_positions = list(chain(*fixed_extruders))+barriersOccupation
        print("Barriers' occupied positions",sorted(occupied_positions))
        lifetimeFactor = 1 #int(log(loop_extrusion_dynamics['lifetime'])/log(10))
        print("Lifetime ",loop_extrusion_dynamics['lifetime'])
        print("Lifetime factor ",lifetimeFactor)
                            
        sys.stdout.flush()        

        print("")
        print("### BEGIN Extruders loading sites ###")
        print("#Extruders_loading_sites")
        if 'ExtrudersLoadingSites' in loop_extrusion_dynamics:
            for site in loop_extrusion_dynamics['ExtrudersLoadingSites']:
                print(site)
        else:
            print("You didn't define any extruders loading sites")
        print("### END Defined Extruders loading sites ###")

        print("### BEGIN Positioning extruders ###")
        offset = 0 
        if 'ExtrudersLoadingSites' in loop_extrusion_dynamics:
            print("Positioning extruders on determined loading sites:")
            extrudersToLoad = []
            for c in range(len(loop_extrusion_dynamics['chrlength'])):
                nextruders    = int(loop_extrusion_dynamics['chrlength'][c]/loop_extrusion_dynamics['separation'])
                nLoadingSites = 0
                print("Number of extruders ",nextruders,"for copy",c+1)
                for site in loop_extrusion_dynamics['ExtrudersLoadingSites']:
                    if offset < site and site <= (offset + loop_extrusion_dynamics['chrlength'][c]):
                        
                        print(offset,site,offset + loop_extrusion_dynamics['chrlength'][c])
                        nLoadingSites += 1
                        new_positions    = [site,site+1]
                        extruders_positions.append(new_positions)
                        occupied_positions = occupied_positions + new_positions
                        print("Occupied positions",sorted(occupied_positions))

                        # Initialise the lifetime of each extruder and the flag to mark it they bumped on a barrier on the right or the left
                        extruders_lifetimes.append(int(0))
                        if lifetimeExponential == 1:
                            lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                            while lifetime == 0:
                                lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                        else:
                            lifetime = loop_extrusion_dynamics['lifetime']
                        extruders_target_lifetimes.append(lifetime)
                        right_bumps.append(int(0))
                        left_bumps.append(int(0))
                        right_bumps_moving.append(int(0))
                        left_bumps_moving.append(int(0))                        
                        if nLoadingSites == nextruders:
                            break
                offset += loop_extrusion_dynamics['chrlength'][c]
                extrudersToLoad.append(nextruders-nLoadingSites)
                print("Still ",extrudersToLoad[-1]," extruders to load over ",nextruders)
                
        else:
            print("Positioning extruders on random loading points:")
            for c in range(len(loop_extrusion_dynamics['chrlength'])): 
                nextruders = int(loop_extrusion_dynamics['chrlength'][c]/loop_extrusion_dynamics['separation'])
                print("Number of extruders ",nextruders,"for copy",c+1)       
                for extruder in range(nextruders):
                    print("Positioning extruder",extruder+1)

                    new_positions = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][c],distances)
                    new_positions[0] = offset + new_positions[0]
                    new_positions[1] = offset + new_positions[1]
                    while (new_positions[0] in occupied_positions) or (new_positions[1] in occupied_positions):
                        new_positions = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][0],distances)
                        if (new_positions[0] in fixed_extruders) or (new_positions[1] in fixed_extruders):
                            print("New positions",new_positions,"discarded")
                            sys.stdout.flush()
                    extruders_positions.append(new_positions)
                    occupied_positions = occupied_positions + new_positions
                    print("Occupied positions",sorted(occupied_positions))

                    # Initialise the lifetime of each extruder and the flag to mark it they bumped on a barrier on the right or the left
                    extruders_lifetimes.append(int(0))
                    if lifetimeExponential == 1:
                        lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                        while lifetime == 0:
                            lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                    else:
                        lifetime = loop_extrusion_dynamics['lifetime']
                    extruders_target_lifetimes.append(lifetime)
                    right_bumps.append(int(0))
                    left_bumps.append(int(0))
                    right_bumps_moving.append(int(0))
                    left_bumps_moving.append(int(0))                    
                offset += loop_extrusion_dynamics['chrlength'][c]
        print("### END Positioning extruders ###")
        extruders_positions = sorted(extruders_positions, key=itemgetter(0))
        print("Initial extruders' positions (All %d)"      % (len(extruders_positions)),extruders_positions)        
        print("Initial extruders' lifetimes (Variable %d)" % (len(extruders_lifetimes)),extruders_lifetimes)
        print("Initial extruders' target lifetimes (Variable %d)" % (len(extruders_target_lifetimes)),extruders_lifetimes)
        if 'ExtrudersLoadingSites' in loop_extrusion_dynamics:
            print("Extruders still to load per chain ",extrudersToLoad)
        #exit(1)
            
        if 'stalling_pairs'  in loop_extrusion_dynamics:
            stalling_pairs_lifetimes  = {}
            stalling_pairs_occupation = {}
            print("#Stalling_pair stalling_time current_stalling_time occupied_flag")
            for pair in loop_extrusion_dynamics['stalling_pairs']:
                stalling_pairs_lifetimes[pair] = 0
                stalling_pairs_occupation[pair] = -1
                print(pair, loop_extrusion_dynamics['stalling_pairs'][pair],stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
            print(stalling_pairs_lifetimes)
            print(stalling_pairs_occupation)

        print("Define the fixed extruders once for all")
        fixed_extruder_number=0
        for particle1,particle2 in fixed_extruders:
            fixed_extruder_number += 1
            print("# fix fixed_LE%i all restrain bond %i  %i %f %f %f" % (fixed_extruder_number,
                                                                          particle1,
                                                                          particle2,
                                                                          loop_extrusion_dynamics['attraction_strength'],
                                                                          loop_extrusion_dynamics['attraction_strength'],
                                                                          loop_extrusion_dynamics['equilibrium_distance']))
            
            lmp.command("fix fixed_LE%i all restrain bond %i  %i %f %f %f" % (fixed_extruder_number,
                                                                              particle1,
                                                                              particle2,
                                                                              loop_extrusion_dynamics['attraction_strength'],
                                                                              loop_extrusion_dynamics['attraction_strength'],
                                                                              loop_extrusion_dynamics['equilibrium_distance']))

            
        print("Defined",fixed_extruder_number,"fixed extruders")

        gene_PolII_occupation = {}
        moving_barriers       = []
        extruderOnBarrierLeftMoving  = []
        extruderOnBarrierRightMoving = []
        if "genes" in loop_extrusion_dynamics:
            for gene in loop_extrusion_dynamics["genes"]:
                gene_PolII_occupation[gene] = {}
                if "PolII_occupancy_per_gene" in loop_extrusion_dynamics:
                    if gene in loop_extrusion_dynamics["PolII_occupancy_per_gene"]:
                        PolII_occupancy = loop_extrusion_dynamics["PolII_occupancy_per_gene"][gene]
                    else:
                        PolII_occupancy = 0.1
                for particle in loop_extrusion_dynamics["genes"][gene]:                    
                    gene_PolII_occupation[gene][particle] = 0
                    randomProb = uniform(0,1)
                    if randomProb <= PolII_occupancy:
                        gene_PolII_occupation[gene][particle] = 1
                        moving_barriers.append(particle)
                        extruderOnBarrierLeftMoving.append(-1)
                        extruderOnBarrierRightMoving.append(-1)      
            print("Moving barriers",moving_barriers)


        lmp.command("compute xu all property/atom xu")
        lmp.command("compute yu all property/atom yu")
        lmp.command("compute zu all property/atom zu")
            
        print("Define the variable extruders")        
        for LES in range(int(run_time/loop_extrusion_dynamics['extrusion_time'])):
            print("### Extrusion round",LES,"###")
            thermo_style="thermo_style   custom   step temp epair emol"
            sys.stdout.flush()
            
            if "PolII_speed" in loop_extrusion_dynamics:
                if LES % loop_extrusion_dynamics["PolII_speed"] == 0:
                    print("Updating moving barriers positions at step",LES)
                    for i in range(len(moving_barriers)):
                        b = moving_barriers[i]
                        for gene in loop_extrusion_dynamics["genes"]:
                            if b in loop_extrusion_dynamics["genes"][gene]:
                                bgene = gene
                                if loop_extrusion_dynamics["genes"][gene][0] <= loop_extrusion_dynamics["genes"][gene][1]:
                                    gene_direction = 1
                                else:
                                    gene_direction = -1
                        # If Poll2 reaches the end of the gene, it is brought again at the beginning
                        if moving_barriers[i] + gene_direction == loop_extrusion_dynamics["genes"][gene][-1]:
                            moving_barriers[i] = loop_extrusion_dynamics["genes"][gene][0]
                        else:
                            moving_barriers[i] += gene_direction
                        extruderOnBarrierLeftMoving[i]  = -1
                        extruderOnBarrierRightMoving[i] = -1
                    print("Updated moving barriers",moving_barriers)
            
            # Update the bond restraint for variable extruders
            variable_extruder_number = 0
            for particle1,particle2 in extruders_positions:                    
                variable_extruder_number += 1
                print("# fix LE%i all restrain bond %i  %i %f %f %f %f" % (variable_extruder_number,
                                                                        particle1,
                                                                        particle2,
                                                                        loop_extrusion_dynamics['attraction_strength'],
                                                                        loop_extrusion_dynamics['attraction_strength'],
                                                                        1.0,
                                                                        loop_extrusion_dynamics['equilibrium_distance']))
                
                lmp.command("fix LE%i all restrain bond %i  %i %f %f %f %f" % (variable_extruder_number,
                                                                                  particle1,
                                                                                  particle2,
                                                                                  loop_extrusion_dynamics['attraction_strength'],
                                                                                  loop_extrusion_dynamics['attraction_strength'],
                                                                                  1.0,
                                                                                  loop_extrusion_dynamics['equilibrium_distance']))
                lmp.command("variable x%i equal c_xu[%i]" % (particle1, particle1))
                lmp.command("variable x%i equal c_xu[%i]" % (particle2, particle2))
                lmp.command("variable y%i equal c_yu[%i]" % (particle1, particle1))
                lmp.command("variable y%i equal c_yu[%i]" % (particle2, particle2))
                lmp.command("variable z%i equal c_zu[%i]" % (particle1, particle1))
                lmp.command("variable z%i equal c_zu[%i]" % (particle2, particle2))
                
                lmp.command("variable xLE%i equal v_x%i-v_x%i" % (variable_extruder_number, particle1, particle2))
                lmp.command("variable yLE%i equal v_y%i-v_y%i" % (variable_extruder_number, particle1, particle2))
                lmp.command("variable zLE%i equal v_z%i-v_z%i" % (variable_extruder_number, particle1, particle2))
                lmp.command("variable dist_%i_%i equal sqrt(v_xLE%i*v_xLE%i+v_yLE%i*v_yLE%i+v_zLE%i*v_zLE%i)" % (particle1,
                                                                                                                 particle2,
                                                                                                                 variable_extruder_number,
                                                                                                                 variable_extruder_number,
                                                                                                                 variable_extruder_number,
                                                                                                                 variable_extruder_number,
                                                                                                                 variable_extruder_number,
                                                                                                                 variable_extruder_number))
                thermo_style += " v_dist_%i_%i" % (particle1, particle2)
            print("Defined",variable_extruder_number,"variable extruders")

            lmp.command("%s" % thermo_style)
            # Doing the LES
            if '1D_run' in loop_extrusion_dynamics:
                lmp.command("run 0")
            else:
                lmp.command("run %i" % loop_extrusion_dynamics['extrusion_time'])

            #exit(1)
            #lmp.command("fix f_unwr all store/state 1 xu yu zu")
            xc_tmp = np.array(lmp.extract_fix("f_unwr",1,2).contents[0:(natoms-2)])
            distances = compute_particles_distance(xc_tmp)
            #print(distances)
                
            # update the lifetime of each extruder
            for extruder in range(len(extruders_positions)):
                extruders_lifetimes[extruder] = extruders_lifetimes[extruder] + 1
            
            # Remove the bond restraints of variable extruders!
            loop_number = 1
            for particle1,particle2 in extruders_positions:
                #print("# unfix LE%i" % (loop_number))
                lmp.command("unfix LE%i" % (loop_number))

                loop_number += 1

            # Update the particles involved in the loop extrusion interaction:
            # decrease the initial_start by one until you get to start
            # increase the initial_stop by one until you get to stop
            extruders_to_relocate = [1]*len(extruders_positions) # 0 if the extruder needs to be relocated and 1 if it doesn't!
            force_extruders_to_relocate = [1]*len(extruders_positions) # 0 if the extruder needs to be relocated and 1 if it doesn't!
            for extruder in range(len(extruders_positions)):
                print("")
                print("#Moving extruder",extruder)
                
                # Keep in memory the current positions
                tmp_extruder = extruders_positions[extruder].copy()
                # Keep in memory the chromosome limits
                total = 0
                start = 1
                nchr  = 0
                for c in loop_extrusion_dynamics['chrlength']:
                    nchr += 1
                    stop  = start + c - 1
                    #print("Chromosome",nchr,"goes from bead",start,"to bead",stop)
                    if start <= extruders_positions[extruder][0] and extruders_positions[extruder][0] <= stop:
                        break                    
                    start = stop + 1                    
                print("Chromosome",nchr,"from bead",start,"to bead",stop,"includes the extruder of position",extruders_positions[extruder])            
                
                # 1. Propose a move of the extruder with probabilities "left_extrusion_rate' or 'right_extrusion_rate'
                # and distinguishing in linear or ring topology
                if loop_extrusion_dynamics['topology'] == "Linear":
                    random_number = uniform(0, 1)
                    if right_bumps[extruder] == 1:
                        random_number = 0
                    if random_number <= float(loop_extrusion_dynamics["left_extrusion_rate"]):       
                        # If the left part reaches the start of the chromosome, put the extruder in another position and re-initialize its lifetime -> Routine to relocate extruder
                        if extruders_positions[extruder][0] > start:
                            extruders_positions[extruder][0] -= 1
                            #print("Propose moving the left arm of the extruder(",random_number,"<=",loop_extrusion_dynamics["left_extrusion_rate"],")",extruder,"from",tmp_extruder[0],"to",extruders_positions[extruder][0])
                        else:
                            #extruders_to_relocate[extruder] = 0 # 0 if the extruder needs to be relocated and 1 if it is doesn't!
                            force_extruders_to_relocate[extruder] = 0 # 0 if the extruder needs to be relocated and 1 if it is doesn't!
                            print("Relocate the extruder",extruder,"because it reached the start of the chain",tmp_extruder[0])
                            
                    random_number = uniform(0, 1)
                    if left_bumps[extruder] == 1:
                        random_number = 0
                    if random_number <= float(loop_extrusion_dynamics["right_extrusion_rate"]):
                        # If the right part reaches the end of the chromosome, put the extruder in another position and re-initialize its lifetime -> Routine to relocate extruder
                        if extruders_positions[extruder][1] < stop:
                            extruders_positions[extruder][1] += 1
                            #print("Propose moving the right arm of the extruder(",random_number,"<=",loop_extrusion_dynamics["right_extrusion_rate"],")",extruder,"from",tmp_extruder[1],"to",extruders_positions[extruder][1])
                        else:
                            #extruders_to_relocate[extruder] = 0 # 0 if the extruder needs to be relocated and 1 if it is doesn't!
                            force_extruders_to_relocate[extruder] = 0 # 0 if the extruder needs to be relocated and 1 if it is doesn't!
                            print("Relocate the extruder",extruder,"because it reached the end of the chain",tmp_extruder[1])                            

                # Move the extruder if it doesn't hit the chromosome limits
                if loop_extrusion_dynamics['topology'] == "Ring":
                    # If the left part reaches the start of the chromosome -> Go to the end
                    random_number = uniform(0, 1)
                    if random_number <= loop_extrusion_dynamics['left_extrusion_rate']:
                        if tmp_extruder[0] > start:
                            extruders_positions[extruder][0] -= 1
                        elif tmp_extruder[0] == start:
                            extruders_positions[extruder][0] = stop
                        elif extruders_positions[extruder][0] == tmp_extruder[1]:
                            extruders_positions[extruder][0] = tmp_extruder[0]
                            
                    # If the right part reaches the end of the chromosome -> Go to start
                    random_number = uniform(0, 1)
                    if random_number <= loop_extrusion_dynamics['right_extrusion_rate']:
                        if tmp_extruder[1] <  stop:
                            extruders_positions[extruder][1] += 1
                        if tmp_extruder[1] == stop:
                            extruders_positions[extruder][1] = start
                        if extruders_positions[extruder][1] == tmp_extruder[0]:
                            extruders_positions[extruder][1] = tmp_extruder[1]
                    #print("Propose moving the extruder",extruder,"from",tmp_extruder,"to",extruders_positions[extruder])
                            
                # 2. If the extruder bumps into another extruder (fixed or variable)
                tmp_extruders_positions = [extruders_positions[x] for x in range(len(extruders_positions)) if x != extruder]

                occupied_positions = list(chain(*fixed_extruders))+list(chain(*tmp_extruders_positions))
                #print("Proposed extruder positions",extruders_positions[extruder])
                #print("Occupied_positions (Extruder excluded)",occupied_positions)

                if   loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "stalling":
                    #Option 1 bring it back
                    if extruders_positions[extruder][0] in occupied_positions:
                        print("The left  part of the extruder bumped into an occupied position: the new position",extruders_positions[extruder][0],"is brought to the previous one",tmp_extruder[0])
                        extruders_positions[extruder][0] = tmp_extruder[0]
                    if extruders_positions[extruder][1] in occupied_positions:
                        print("The right part of the extruder bumped into an occupied position: the new position",extruders_positions[extruder][1],"is brought to the previous one",tmp_extruder[1])
                        extruders_positions[extruder][1] = tmp_extruder[1]
                        
                elif loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "relocating":
                    #Option 2 put it in another position and re-initialize its lifetime -> Routine to relocate extruder
                    if extruders_positions[extruder][0] in occupied_positions:
                        print("The left arm of the extruder",extruder,"bumped into an occupied position: the new position",extruders_positions[extruder][0],"it is relocated to a new random position")
                        extruders_to_relocate[extruder] = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                    if extruders_positions[extruder][1] in occupied_positions:
                        print("The right arm of the extruder",extruder,"bumped into an occupied position: the new position",extruders_positions[extruder][1],"it is relocated to a new random position")
                        extruders_to_relocate[extruder] = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!

                elif loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "crossing":
                    #Option 3 extruders can cross each other
                    if extruders_positions[extruder][0] in occupied_positions:
                        print("The left  arm of the extruder",extruder,"bumped into an occupied position, but they can cross each other")
                    if extruders_positions[extruder][1] in occupied_positions:
                        print("The right arm of the extruder",extruder,"bumped into an occupied position, but they can cross each other")

                # 3.5 if the extruder reaches a PolII moving barrier, the extruder will stop if it is in the direction of the transcription
                if extruders_positions[extruder][0] in moving_barriers:                    
                    perm = 0.2
                    # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability                    
                    # If the barrier is on the left of the extruders, which is extruding contrary to the chain index, we have to re-put the extruder forwards
                    print("Found a moving barrier coming from the right at monomer %d with permeability %f" % (extruders_positions[extruder][0],perm))
                    # If the extruder was already stopped right_bumps[extruder] == 1, we have to block it
                    randomProb = uniform(0,1)
                    if right_bumps_moving[extruder] == 1:
                        print("The extruder was already stopped at this barrier, it will stay here until the end of its lifetime ",extruders_target_lifetimes[extruder])
                        randomProb = 1.
                    if randomProb > perm:
                        # Option 1: The extruder stops at the barrier
                        right_bumps_moving[extruder] = 1
                        extruders_to_relocate[extruder]      = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                        # Now, if the extruders can cross each other, they tend to concentrate at barriers, to avoid that we forbid an extruder to stall at an occupied barrier and relocate it
                        if loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "crossing":
                            print("ExtrudeOnbarrier",extruderOnBarrierRightMoving)
                            print("Barriers",moving_barriers)
                            print("Index of the barrier",moving_barriers.index(extruders_positions[extruder][0]))
                            if extruderOnBarrierRightMoving[moving_barriers.index(extruders_positions[extruder][0])] != extruder and extruderOnBarrierRightMoving[moving_barriers.index(extruders_positions[extruder][0])] != -1:
                                print("When the extruders can cross each other, they tend to concentrate at barriers.")
                                print("To avoid that we forbid an extruder to stall at an occupied barrier and relocate it.")
                                print("Accordingly I will relocate extruder",extruder)
                                extruders_to_relocate[extruder]      = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                            else:
                                extruderOnBarrierRightMoving[moving_barriers.index(extruders_positions[extruder][0])] = extruder
                        print("The left  part of the extruder",extruder,"bumped in a moving barrier: the new position",extruders_positions[extruder][0],"is brought to the previous one",tmp_extruder[0])
                        extruders_positions[extruder][0] = tmp_extruder[0]
                    #exit(1)
                        
                if extruders_positions[extruder][1] in moving_barriers:
                    perm = 0.2
                    # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability
                    # If the barrier is on the right of the extruders, which is extruding with the chain index, we have to re-put the extruder backwards
                    print("Found a barrier coming from the left at monomer %d with permeability %f" % (extruders_positions[extruder][1],perm))
                    # If the extruder was already stopped left_bumps[extruder] == 1, we have to block it
                    randomProb = uniform(0,1)
                    if left_bumps_moving[extruder] == 1:
                        print("The extruder was already stopped at this barrier, it will stay here until the end of its lifetime ",extruders_target_lifetimes[extruder])
                        randomProb = 1
                    if randomProb > perm:
                        # Option 1: The extruder stops at the barrier
                        left_bumps_moving[extruder] = 1
                        # Avoid to relocate the extruder!
                        extruders_to_relocate[extruder] = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                        # Now, if the extruders can cross each other, they tend to concentrate at barriers, to avoid that we forbid an extruder to stall at an occupied barrier and relocate it
                        if loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "crossing":
                            print("ExtrudeOnbarrier",extruderOnBarrierLeftMoving)
                            print("Barriers",moving_barriers)
                            print("Index of the barrier",moving_barriers.index(extruders_positions[extruder][1]))
                            if extruderOnBarrierLeftMoving[moving_barriers.index(extruders_positions[extruder][1])] != extruder and extruderOnBarrierLeftMoving[moving_barriers.index(extruders_positions[extruder][1])] != -1:
                                print("When the extruders can cross each other, they tend to concentrate at barriers.")
                                print("To avoid that we forbid an extruder to stall at an occupied barrier and relocate it.")
                                print("Accordingly I will relocate extruder",extruder)
                                extruders_to_relocate[extruder]      = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                            else:
                                extruderOnBarrierLeftMoving[moving_barriers.index(extruders_positions[extruder][1])] = extruder
                        print("The right part of the extruder",extruder,"bumped in a barrier: the new position",extruders_positions[extruder][1],"is brought to the previous one",tmp_extruder[1])
                        extruders_positions[extruder][1] = tmp_extruder[1]

                        
                # 3. if the extruder reaches a barrier it is stop or not depending on the permeability of the barrier
                if   loop_extrusion_dynamics["loop_extruder_barrier_encounter_rule"] == "stalling":

                    if extruders_positions[extruder][0] in loop_extrusion_dynamics['barriers']:
                        perm = loop_extrusion_dynamics['barriers_right_permeability'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])]
                        if 'lifetimeAtBarriersRight' in loop_extrusion_dynamics:
                            barrierLifetimeRight = loop_extrusion_dynamics['lifetimeAtBarriersRight'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])]
                        # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability
                        # If the barrier is on the left of the extruders, which is extruding contrary to the chain index, we have to re-put the extruder forwards
                        print("Found a barrier coming from the right at monomer %d with permeability %f" % (extruders_positions[extruder][0],perm))
                        # If the extruder was already stopped right_bumps[extruder] == 1, we have to block it
                        randomProb = uniform(0,1)
                        if right_bumps[extruder] == 1:
                            print("The extruder was already stopped at this barrier, it will stay here until the end of its lifetime ",extruders_target_lifetimes[extruder])
                            if 'stalling' in loop_extrusion_dynamics:
                                extruders_positions[extruder][0] = tmp_extruder[0]
                                extruders_positions[extruder][1] = tmp_extruder[1]
                            randomProb = 1.
                        if randomProb > perm:
                            # Option 1: The extruder stops at the barrier
                            if right_bumps[extruder] != 1:
                                tmpLifetime = extruders_lifetimes[extruder] 
                                extruders_target_lifetimes[extruder] = tmpLifetime + barrierLifetimeRight
                            right_bumps[extruder] = 1
                            extruders_to_relocate[extruder]      = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                            # Now, if the extruders can cross each other, they tend to concentrate at barriers, to avoid that we forbid an extruder to stall at an occupied barrier and relocate it
                            if loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "crossing":
                                print("ExtrudeOnbarrier",extruderOnBarrierRight)
                                print("Barriers",loop_extrusion_dynamics['barriers'])
                                print("Index of the barrier",loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0]))
                                if extruderOnBarrierRight[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])] != extruder and extruderOnBarrierRight[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])] != -1:
                                    print("When the extruders can cross each other, they tend to concentrate at barriers.")
                                    print("To avoid that we forbid an extruder to stall at an occupied barrier and relocate it.")
                                    print("Accordingly I will relocate extruder",extruder)
                                    extruders_to_relocate[extruder]      = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                                else:
                                    extruderOnBarrierRight[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])] = extruder
                            print("The left  part of the extruder",extruder,"bumped in a barrier: the new position",extruders_positions[extruder][0],"is brought to the previous one",tmp_extruder[0])
                            extruders_positions[extruder][0] = tmp_extruder[0]

                    if extruders_positions[extruder][1] in loop_extrusion_dynamics['barriers']:
                        perm = loop_extrusion_dynamics['barriers_left_permeability'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])]
                        if 'lifetimeAtBarriersLeft' in loop_extrusion_dynamics:
                            barrierLifetimeLeft = loop_extrusion_dynamics['lifetimeAtBarriersLeft'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])]
                        # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability
                        # If the barrier is on the right of the extruders, which is extruding with the chain index, we have to re-put the extruder backwards
                        print("Found a barrier coming from the left at monomer %d with permeability %f" % (extruders_positions[extruder][1],perm))
                        # If the extruder was already stopped left_bumps[extruder] == 1, we have to block it
                        randomProb = uniform(0,1)
                        if left_bumps[extruder] == 1:
                            print("The extruder was already stopped at this barrier, it will stay here until the end of its lifetime ",extruders_target_lifetimes[extruder])
                            if 'stalling' in loop_extrusion_dynamics:
                                extruders_positions[extruder][0] = tmp_extruder[0]
                                extruders_positions[extruder][1] = tmp_extruder[1]
                            randomProb = 1
                        if randomProb > perm:
                            # Option 1: The extruder stops at the barrier
                            if left_bumps[extruder] != 1:
                                tmpLifetime = extruders_lifetimes[extruder] 
                                extruders_target_lifetimes[extruder] = tmpLifetime + barrierLifetimeLeft
                            left_bumps[extruder] = 1
                            # Avoid to relocate the extruder!
                            extruders_to_relocate[extruder] = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                            # Now, if the extruders can cross each other, they tend to concentrate at barriers, to avoid that we forbid an extruder to stall at an occupied barrier and relocate it
                            if loop_extrusion_dynamics["loop_extruders_encounter_rule"] == "crossing":
                                print("ExtrudeOnbarrier",extruderOnBarrierLeft)
                                print("Barriers",loop_extrusion_dynamics['barriers'])
                                print("Index of the barrier",loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1]))
                                if extruderOnBarrierLeft[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])] != extruder and extruderOnBarrierLeft[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])] != -1:
                                    print("When the extruders can cross each other, they tend to concentrate at barriers.")
                                    print("To avoid that we forbid an extruder to stall at an occupied barrier and relocate it.")
                                    print("Accordingly I will relocate extruder",extruder)
                                    extruders_to_relocate[extruder]      = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                                else:
                                    extruderOnBarrierLeft[loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])] = extruder
                            print("The right part of the extruder",extruder,"bumped in a barrier: the new position",extruders_positions[extruder][1],"is brought to the previous one",tmp_extruder[1])
                            extruders_positions[extruder][1] = tmp_extruder[1]


                elif loop_extrusion_dynamics["loop_extruder_barrier_encounter_rule"] == "relocating":
                    if extruders_positions[extruder][0] in loop_extrusion_dynamics['barriers']:
                        perm = loop_extrusion_dynamics['barriers_right_permeability'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][0])]
                        # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability
                        # If the barrier is on the left of the extruders, which is extruding contrary to the chain index, we have to re-put the extruder forwards
                        print("Found a barrier coming from the right at monomer %d with permeability %f" % (extruders_positions[extruder][0],perm))
                        if uniform(0,1) > perm:
                            print("The left  part of the extruder",extruder,"bumped in a barrier and it will be relocated")
                            extruders_to_relocate[extruder] = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!

                    if extruders_positions[extruder][1] in loop_extrusion_dynamics['barriers']:
                        perm = loop_extrusion_dynamics['barriers_left_permeability'][loop_extrusion_dynamics['barriers'].index(extruders_positions[extruder][1])]
                        # If the extruder tries to overcome a barrier we stop it with a probability > than the permeability
                        # If the barrier is on the right of the extruders, which is extruding with the chain index, we have to re-put the extruder backwards
                        print("Found a barrier coming from the left at monomer %d with permeability %f" % (extruders_positions[extruder][1],perm))
                        if uniform(0,1) > perm:
                            print("The right part of the extruder",extruder,"bumped in a barrier and it will be relocated")
                            extruders_to_relocate[extruder] = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!                            

                # 3b. If the extruder connects two stalling_pairs it will stay there for the life of the stalling pair
                if 'stalling_pairs'  in loop_extrusion_dynamics:
                    for pair in loop_extrusion_dynamics['stalling_pairs']:
                        if (tmp_extruder[0] == pair[0] and tmp_extruder[1] == pair[1]) or (tmp_extruder[0] == pair[1] and tmp_extruder[1] == pair[0]):
                            print("Extruder",extruder,"found a potential stalling pair",pair,loop_extrusion_dynamics['stalling_pairs'][pair])
                            # If the stalling pair is already occupied, just leave
                            if stalling_pairs_occupation[pair] != -1 and stalling_pairs_occupation[pair] != extruder:
                                print("But extruder",extruder,"found an occupied stalling pair",pair,stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
                                print("Accordingly, it will continue extruding")
                            else:
                                if stalling_pairs_lifetimes[pair] == loop_extrusion_dynamics['stalling_pairs'][pair]:
                                    print("Extruder",extruder,"stalled in a pair for the target lifetime")
                                    print(pair, loop_extrusion_dynamics['stalling_pairs'][pair],stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
                                    print("It will keep extruding now!")
                                    stalling_pairs_lifetimes[pair]  = 0
                                    stalling_pairs_occupation[pair] = -1
                                    print("The lifetime and occupation of the stalling pairs are re-initialized")
                                    print(pair,"lifetime",stalling_pairs_lifetimes[pair],"target-lifetime",loop_extrusion_dynamics['stalling_pairs'][pair],"occupation",stalling_pairs_occupation[pair])
                                else:
                                    print("Extruder",extruder,"stalled in a pair")
                                    print("#Stalling_pair stalling_time current_stalling_time occupied_flag")
                                    print(pair, loop_extrusion_dynamics['stalling_pairs'][pair],stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
                                    sys.stdout.flush()
                                    stalling_pairs_occupation[pair] = extruder
                                    print("Pause the extruder in this position")
                                    extruders_positions[extruder][0] = tmp_extruder[0]
                                    extruders_positions[extruder][1] = tmp_extruder[1]
                                    stalling_pairs_lifetimes[pair] = stalling_pairs_lifetimes[pair] + 1
                                    print("Avoid to relocate the extruder")
                                    extruders_to_relocate[extruder] = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!

                        """
                        if (extruders_positions[extruder][0] == pair[0] and extruders_positions[extruder][1] == pair[1]) or (extruders_positions[extruder][0] == pair[1] and extruders_positions[extruder][1] == pair[0]):
                            if stalling_pairs_occupation[pair] != -1 and stalling_pairs_occupation[pair] != extruder:
                                continue
                            else:
                                if stalling_pairs_lifetimes[pair] == loop_extrusion_dynamics['stalling_pairs'][pair]:
                                    print("Extruder",extruder,"stalled in a pair for the target lifetime")
                                    print(pair, loop_extrusion_dynamics['stalling_pairs'][pair],stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
                                    print("It will keep extruding now!")
                                    stalling_pairs_lifetimes[pair] = 0
                                else:
                                    print("Extruder",extruder,"stalled in a pair")
                                    print("#Stalling_pair stalling_time current_stalling_time occupied_flag")
                                    print(pair, loop_extrusion_dynamics['stalling_pairs'][pair],stalling_pairs_lifetimes[pair],stalling_pairs_occupation[pair])
                                    sys.stdout.flush()
                                    stalling_pairs_occupation[pair] = extruder
                                    stalling_pairs_lifetimes[pair]   = stalling_pairs_lifetimes[pair] + 1 
                                    print("Pause the extruder in this position")
                                    extruders_positions[extruder][0] = tmp_extruder[0]
                                    extruders_positions[extruder][1] = tmp_extruder[1]
                                    stalling_pairs_lifetimes[pair] = stalling_pairs_lifetimes[pair] + 1
                                    print("Avoid to relocate the extruder")
                                    extruders_to_relocate[extruder] = 1  # 0 if the extruder needs to be relocated and 1 if it doesn't!                              
                        """
                # 3c. If the extruder connects two switchOn_pairs it will apply an attractive interaction between them as in the compartmentalization case,
                # So the structure of loop_extrusion_dynamics['switchOn_pairs'] is the same of compartmentalization['interactions']
                if 'switchOn_pairs'  in loop_extrusion_dynamics:
                    for pair in loop_extrusion_dynamics['switchOn_pairs']:
                        #{(1196,1198) : ["attraction",4.0000000000]}
                        t1      = pair[0]
                        t2      = pair[1]
                        epsilon = loop_extrusion_dynamics['switchOn_pairs'][pair][1]
                        sigma = 1.0
                        rc = sigma * 2.5
                        lmp.command("pair_coeff %s %s 2 %s %s %s" % (t1,t2,epsilon,sigma,rc))

                # XXX. If the new bond is larger than 100nm
                #if loop_extrusion_dynamics['max_distance_to_create']:

                # 4. If the extruder reached its lifetime, put it in another position and re-initialize its lifetime -> Routine to relocate extruder
                if extruders_lifetimes[extruder] >= extruders_target_lifetimes[extruder]:
                    extruders_to_relocate[extruder] = 0  # 0 if the extruder needs to be relocated and 1 if it doesn't!

                # Routine to relocate extruder
                if extruders_to_relocate[extruder] == 0 or force_extruders_to_relocate[extruder] == 0:  # 0 if the extruder needs to be relocated and 1 if it doesn't!
                    print("Relocating extruder ",extruder," lifetime ",extruders_lifetimes[extruder]," Target-lifetime ",extruders_target_lifetimes[extruder])                    
                    tmp_extruders_positions = [extruders_positions[x] for x in range(len(extruders_positions)) if x != extruder]
                    occupied_positions = list(chain(*fixed_extruders))+list(chain(*tmp_extruders_positions))+barriersOccupation
                    print("Occupied_positions (Extruder excluded)",sorted(occupied_positions))

                    if 'ExtrudersLoadingSites' in loop_extrusion_dynamics:
                        offset = 0
                        for c in range(len(loop_extrusion_dynamics['chrlength'])):                                
                            if c != (nchr-1):
                                continue
                            nLoadingSites = 0
                            sites = []
                            print(sites)
                            for site in loop_extrusion_dynamics['ExtrudersLoadingSites']:
                                if offset < site and site <= (offset + loop_extrusion_dynamics['chrlength'][c]):
                                    sites.append(site)
                            sites = sample(sites, len(sites))
                            print(sites)

                            for site in sites:
                                print(offset,site,offset + loop_extrusion_dynamics['chrlength'][c])
                                new_positions    = [site,site+1]                                
                                extruders_positions[extruder] = new_positions
                                if (extruders_positions[extruder][0] in occupied_positions) or (extruders_positions[extruder][1] in occupied_positions):
                                    print("One of the proposed random positions",extruders_positions[extruder],"is occupied")
                                    continue
                                nLoadingSites += 1
                                break
                            if nLoadingSites == 0:
                                print("NOTE: not enough loading determined loading sites to load this extruders: Using a random one!")                                
                                extruders_positions[extruder]    = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][nchr-1],distances)
                                extruders_positions[extruder][0] = extruders_positions[extruder][0] + start
                                extruders_positions[extruder][1] = extruders_positions[extruder][1] + start
                                print("Proposed random position",extruders_positions[extruder])
                    
                                while (extruders_positions[extruder][0] in occupied_positions) or (extruders_positions[extruder][1] in occupied_positions):
                                    print("One of the proposed random positions",extruders_positions[extruder],"is occupied")
                                    extruders_positions[extruder]    = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][nchr-1],distances)
                                    extruders_positions[extruder][0] = extruders_positions[extruder][0] + start
                                    extruders_positions[extruder][1] = extruders_positions[extruder][1] + start
                                    print("Proposed random position",extruders_positions[extruder])
                                    print("")
                            offset += loop_extrusion_dynamics['chrlength'][c]
                    else:
                        extruders_positions[extruder]    = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][nchr-1],distances)
                        extruders_positions[extruder][0] = extruders_positions[extruder][0] + start
                        extruders_positions[extruder][1] = extruders_positions[extruder][1] + start
                        print("Proposed random position",extruders_positions[extruder])
                    
                        #and (start <= extruders_positions[extruder][1] and extruders_positions[extruder][1] <= stop)
                        #and (start <= extruders_positions[extruder][0] and extruders_positions[extruder][0] <= stop):
                        while (extruders_positions[extruder][0] in occupied_positions) or (extruders_positions[extruder][1] in occupied_positions):
                            print("One of the proposed random positions",extruders_positions[extruder],"is occupied")
                            extruders_positions[extruder]    = draw_loop_extruder_loading_site(loop_extrusion_dynamics['chrlength'][nchr-1],distances)
                            extruders_positions[extruder][0] = extruders_positions[extruder][0] + start
                            extruders_positions[extruder][1] = extruders_positions[extruder][1] + start
                            print("Proposed random position",extruders_positions[extruder])

                    # Re-initialise the lifetime of the extruder
                    extruders_lifetimes[extruder] = 0
                    right_bumps[extruder] = 0
                    left_bumps[extruder] = 0
                    right_bumps_moving[extruder] = 0
                    left_bumps_moving[extruder] = 0                    
                    if lifetimeExponential == 1:
                        lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                        while lifetime == 0:
                            lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                    else:
                        lifetime = loop_extrusion_dynamics['lifetime']
                    extruders_target_lifetimes[extruder] = lifetime

                    # Re-initialise the occupation of barriers on the left, if the extruder was stalling there
                    for i in range(len(extruderOnBarrierLeft)):
                        if extruderOnBarrierLeft[i] == extruder:                            
                            extruderOnBarrierLeft[i] = -1
                            print("Barrier",loop_extrusion_dynamics['barriers'][i],"is now free of extruders at the left!")
                    # Re-initialise the occupation of barriers on the right, if the extruder was stalling there
                    for i in range(len(extruderOnBarrierRight)):
                        if extruderOnBarrierRight[i] == extruder:                            
                            extruderOnBarrierRight[i] = -1
                            print("Barrier",loop_extrusion_dynamics['barriers'][i],"is now free of extruders at the right!")
                    # Re-initialise the occupation of barriers on the left, if the extruder was stalling there
                    for i in range(len(extruderOnBarrierLeftMoving)):
                        if extruderOnBarrierLeftMoving[i] == extruder:                            
                            extruderOnBarrierLeftMoving[i] = -1
                            print("Barrier",moving_barriers[i],"is now free of extruders at the left!")
                    # Re-initialise the occupation of barriers on the right, if the extruder was stalling there
                    for i in range(len(extruderOnBarrierRightMoving)):
                        if extruderOnBarrierRightMoving[i] == extruder:                            
                            extruderOnBarrierRightMoving[i] = -1
                            print("Barrier",loop_extrusion_dynamics['barriers'][i],"is now free of extruders at the right!")

                    #extruders_positions = sorted(extruders_positions, key=itemgetter(0))
                    #print("Extruder",extruder,"relocated from",tmp_extruder,"to",extruders_positions[extruder])

            print("Extruders positions at step",LES,extruders_positions)
            print("Extruders lifetimes at step",LES,extruders_lifetimes)
            print("Extruders target lifetimes at step",LES,extruders_target_lifetimes)
            print("Extruder on barriers left", LES,extruderOnBarrierLeft)
            print("Extruder on barriers right",LES,extruderOnBarrierRight)
            print("Left bumps at step",LES,left_bumps)
            print("Right bumps at step",LES,right_bumps)
            print("Left bumps moving barriers at step",LES,left_bumps_moving)            
            print("Right bumps moving barriers at step",LES,right_bumps_moving)
            sys.stdout.flush()

            if 'ExtrudersLoadingSites' in loop_extrusion_dynamics and sum(extrudersToLoad) != 0:
                print("Try to position the missing extruders at determined loading sites")
                offset = 0
                occupied_positions = list(chain(*fixed_extruders))+list(chain(*extruders_positions))+barriersOccupation
                extrudersToLoadTmp = []
                for c in range(len(loop_extrusion_dynamics['chrlength'])):        
                    nextruders    = extrudersToLoad[c]
                    if nextruders == 0:
                        extrudersToLoadTmp.append(0)
                        continue
                    nLoadingSites = 0
                    print("Number of extruders ",nextruders,"for copy",c+1)
                    sites = []
                    print(sites)
                    for site in loop_extrusion_dynamics['ExtrudersLoadingSites']:
                        if offset < site and site <= (offset + loop_extrusion_dynamics['chrlength'][c]):
                            sites.append(site)
                    sites = sample(sites, len(sites))
                    print(sites)
                    for site in sites:
                        print(offset,site,offset + loop_extrusion_dynamics['chrlength'][c])
                        nLoadingSites += 1
                        new_positions    = [site,site+1]
                        extruders_positions.append(new_positions)
                        occupied_positions = occupied_positions + new_positions
                        print("Occupied positions",sorted(occupied_positions))
                        
                        # Initialise the lifetime of each extruder and the flag to mark it they bumped on a barrier on the right or the left
                        extruders_lifetimes.append(int(0))
                        if lifetimeExponential == 1:
                            lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                            while lifetime == 0:
                                lifetime = lifetimeFactor*np.random.exponential(loop_extrusion_dynamics['lifetime']/lifetimeFactor, size=1)[0]
                        else:
                            lifetime = loop_extrusion_dynamics['lifetime']
                        extruders_target_lifetimes.append(lifetime)
                        right_bumps.append(int(0))
                        left_bumps.append(int(0))
                        right_bumps_moving.append(int(0))
                        left_bumps_moving.append(int(0))                        
                        if nLoadingSites == nextruders:
                            break
                    offset += loop_extrusion_dynamics['chrlength'][c]
                    extrudersToLoadTmp.append(nextruders-nLoadingSites)
                    print("Still ",extrudersToLoadTmp[-1]," extruders to load over ",nextruders)                    
                    print("")
                extrudersToLoad = extrudersToLoadTmp.copy()
                print("Extruders still to load per chain ",extrudersToLoad)
            sys.stdout.flush()            
            
    if not compartmentalization and not loop_extrusion_dynamics:
        lmp.command("run %i" % run_time)
        lmp.command("write_data final_conformation.txt nocoeff")

    ### Put here the creationg of a pickle with the complete trajectory ###
    if to_dump:
        lmp.command("undump 1")
        #lmp.command("dump    1       all    custom    %i   %sloop_extrusion_MD_*.XYZ  id  xu yu zu" % (to_dump,lammps_folder))
        lmp.command("dump    1       all    custom    %i   %sloop_extrusion_MD_*.lammpstrj  id  xu yu zu" % (to_dump,lammps_folder))
    # Post-processing analysis
    # Save coordinates
    #for time in range(0,runtime,to_dump):
    #    xc.append(np.array(read_trajectory_file("%s/loop_extrusion_MD_%s.XYZ" % (lammps_folder, time))))
        
    #xc.append(np.array(lmp.gather_atoms("xu",1,3)))
            
    lmp.close()    
        
    return (kseed, result, init_conf)

### The function for Giorgetti's modelling Giorgetti L,..., Heard E., Cell 2014 ###
# This splits the lammps calculations on different processors:
def giorgettis_modelling(
        target_matrix,
        lammps_folder,
        run_time,
        nreplicas,
        totmodels,
        nparticles,
        alpha,
        B0,
        rc=1.5,
        connectivity="FENEspecial",
        initial_seed=0,
        n_cpus=2,
        gamma=CONFIG.gamma,
        timestep=0.012,
        reset_timestep=0,
        confining_environment=None,
        to_dump=100000,
        hide_log=False,
        autocorrelation_time = None):

    """
    :param None outfile: store result in outfile
    :param 1 n_cpus: number of CPUs to use.
    :param False restart_path: path to files to restore LAMMPs session (binary)
    :param 10 store_n_steps: Integer with number of steps to be saved if 
        restart_file != False
    :param False useColvars: True if you want the restrains to be loaded by colvars

    :returns: a Tadphys models dictionary

    """
    
    if initial_seed:
        seed(initial_seed)

    # Open and load target matrix
    input_matrix = zeros((nparticles,nparticles),dtype = "float")
    sigma        = zeros((nparticles,nparticles),dtype = "float")
    masked = []
    fp = open(target_matrix, "r")
    nl=0
    for line in fp.readlines():

        line = line.strip().split()
        if line[0] == "#" and len(line) > 2:
            for m in range(2,len(line)):
                masked.append(int(line[m]))
            continue

        if line[0] == "#":
            continue

        # Input is a tabulate (.tab) matrix
        d = sqrt((int(line[0])-int(line[1]))*(int(line[0])-int(line[1])))
        
        input_matrix[int(line[0])][int(line[1])] = float(line[2])
        if d <= 1:
            input_matrix[int(line[0])][int(line[1])] = 0
            sigma[int(line[0])][int(line[1])]        = 1.0
            continue
            
        try:
            sigma[int(line[0])][int(line[1])]        = float(line[3])
        except:
            sigma[int(line[0])][int(line[1])]        = 1.0
        if sigma[int(line[0])][int(line[1])] == 0:
            sigma[int(line[0])][int(line[1])] = 1.0

        # Set to zero the intra-bins interactions:
        #input_matrix[int(nl)][int(nl)] = 0.0
            
        nl += 1
    print("Masked bins",masked)
    
    # Normalise the input matrix by the total number or reads
    #input_matrix = input_matrix / input_matrix.sum()
    
    # Normalize the input matrix by the typical number of contacts of
    # two regions to be considered always in contact (Z). To estimate
    # Z from experimental data, we considered the average number of
    # contacts between neirest neighbors bins (genomic distance L=1).
    #cnt    = zeros(nparticles)
    #c_vs_L = zeros(nparticles)
    #c_max  = zeros(nparticles)
    #for i in range(nparticles):
    #    for j in range(i,nparticles):
    #        if i in masked or j in masked:
    #            continue
    #        d = j-i            
    #        c_vs_L[d] += input_matrix[i][j]
    #        if input_matrix[i][j] > c_max[d]:
    #            c_max[d] = input_matrix[i][j]
    #        cnt[d]    += 1                       
            

    #Z            = c_max[2]
    #if Z != 0:
    #    input_matrix = input_matrix / Z
    # Put the nearest neighbors always in contact
    #for i in range(nparticles):
    #    input_matrix[i][i] = 0.0
    #    try:
    #        input_matrix[i][i+1] = 1.0
    #        input_matrix[i+1][i] = 1.0
    #    except:
    #        pass
    fp_output = open("input_matrix.txt","w")
    for i in range(nparticles):
        for j in range(nparticles):
            fp_output.write("%d %d %f %f\n" % (i,j,input_matrix[i][j],sigma[i][j]))
    fp_output.close()

    print(confining_environment)
    # Generate nreplicas initial conformations as random walks
    for replica in range(1,nreplicas+1):
        replica_folder = lammps_folder + 'replica_' + str(replica) + '/'
        if not os.path.exists(replica_folder):
            os.makedirs(replica_folder)
        initial_conformation = "%sInitial_conformation_replica_%s.txt" % (replica_folder,replica)
        if not os.path.exists(initial_conformation):
            generate_chromosome_random_walks_conformation ( chromosome_particle_numbers=[nparticles], 
                                                            confining_environment=confining_environment,
                                                            particle_radius=0.8,
                                                            seed_of_the_random_number_generator=randint(1,1000000),
                                                            number_of_conformations=1,
                                                            outfile = initial_conformation,
                                                            atom_types = nparticles+1)

    if nparticles <= 2000:
        neighbor="3.0 nsq"
    else:
        neighbor="0.3 bin"
    interactions = zeros((nparticles,nparticles),dtype = "float")
    nreplicas_ACT = 3
    if autocorrelation_time is None:

        # Estimate the run_time from the Ree and Rg auto-correlation time (ACT).
        # We will use 3 time the maximum of the two ACT.
        # To estimate the 2 ACT (Ree and Rg) we will first do 10 long runs of 10MLN steps each
        # with no attractive interaction. We will consider as the ACT the average of the 3 ACTs

        # Produce the population of models
        replica_seeds = []        
        for replica in range(nreplicas_ACT):
            replica_seeds.append(replica+1+initial_seed)
        pool = multiprocessing.Pool(processes=n_cpus)
    
        results = []
        def collect_result(result):
            results.append((result[0], result[1], result[2]))

        start_time = time.time()            
        jobs = {}

        print(confining_environment)
        for replica_id, replica in enumerate(replica_seeds):
            replica_folder = lammps_folder + 'replica_' + str(replica) + '_estimate_of_ACTs/'
            if not os.path.exists(replica_folder):
                os.makedirs(replica_folder)
            ini_conf = "%sInitial_conformation_replica_%s.txt" % (replica_folder,replica)
            generate_chromosome_random_walks_conformation (chromosome_particle_numbers=[nparticles], 
                                                           confining_environment=confining_environment,
                                                           particle_radius=0.5,
                                                           seed_of_the_random_number_generator=randint(1,1000000),
                                                           number_of_conformations=1,
                                                           outfile = ini_conf,
                                                           atom_types = nparticles+1)

        for replica_id, replica in enumerate(replica_seeds):
            replica_folder = lammps_folder + 'replica_' + str(replica) + '_estimate_of_ACTs/'
            ini_conf = "%sInitial_conformation_replica_%s.txt" % (replica_folder,replica)
            partition = {}
            radii     = {}
            interactions_parameters = {}
            for i in range(1,nparticles+1):
                partition[i] = i
                radii[i]     = 0.5
            for i in range(1,nparticles+1):
                for j in range(i+1,nparticles+1):
                    pair = (i,j)                
                    interactions_parameters[pair] = ['repulsion',1.0] 
            compartmentalization = {
                'partition'    : partition,
                'radii'        : radii,
                'interactions' : interactions_parameters,
                'runtime'      : 100000000,
                'distances'    : int(run_time/50),
                'gyration'     : 100,
                'Ree'          : 100,
            }
            tethering=False
            minimize=True
            compress_with_pbc=None
            compress_without_pbc=None
            initial_relaxation=None
            confining_environment=None
            steering_pairs=None
            time_dependent_steering_pairs=None
            loop_extrusion_dynamics_OLD=None
            loop_extrusion_dynamics=None            
            pbc=False
            keep_restart_out_dir2=None
            restart_file=False
            model_path=False
            store_n_steps=None
            useColvars=False
            chromosome_particle_numbers = [nparticles]
            timestep=timestep
            reset_timestep=reset_timestep
            
            
            #print(replica_folder)
            jobs[replica] = partial(abortable_worker, run_lammps, timeout=3600*24*30,
                                    failedSeedLog=["runLog.txt", replica])
            #print(jobs)
            pool.apply_async(jobs[replica],
                             args=(replica, replica_folder, run_time,
                                   ini_conf, connectivity,
                                   neighbor,
                                   tethering, minimize,
                                   compress_with_pbc, compress_without_pbc,
                                   initial_relaxation,
                                   restrained_dynamics,
                                   confining_environment,
                                   steering_pairs,
                                   time_dependent_steering_pairs,
                                   compartmentalization,
                                   loop_extrusion_dynamics,
                                   to_dump, pbc, hide_log,
                                   gamma,timestep,
                                   reset_timestep,
                                   chromosome_particle_numbers,
                                   keep_restart_out_dir2,
                                   restart_file,
                                   model_path,
                                   store_n_steps,
                                   useColvars,), callback=collect_result)
            
        pool.close()
        pool.join()
        print("--- %.2f seconds --- ACTs estimate" % (time.time() - start_time))

        # Compute the Ree and Rg ACTs
        Ree_ACT = 0
        Rg_ACT  = 0
        for replica in range(1,nreplicas_ACT+1):
            Ree_ACT += estimate_Ree_ACT(lammps_folder,replica)
            Rg_ACT  += estimate_Rg_ACT(lammps_folder,replica)
            print("Replica %i: tau_Ree = %f , tau_Rg = %f" % (replica,Ree_ACT,Rg_ACT))
        Ree_ACT /= nreplicas_ACT
        Rg_ACT  /= nreplicas_ACT
        print("Average: tau_Ree = %f , tau_Rg = %f" % (Ree_ACT,Rg_ACT))
        
        # To have 5,000 models generated each time the run_time is:
        # 3 times the maximum of Ree_ACT and Rg_ACT to allow statistical independence of the each model along the same trajectory
        # times 5,000/nreplicas to collect 5,000 models combining all replicas
        autocorrelation_time = np.array([Ree_ACT,Rg_ACT]).max() 

    run_time  = int((totmodels/nreplicas) * 10.0 * int(autocorrelation_time))
    #run_time  = int((totmodels/nreplicas) * 5.0 * int(autocorrelation_time))
    save_time = int(run_time/(totmodels/nreplicas))

    print("The estimated auto-correlation time is %d timesteps" % (autocorrelation_time))
    print("The run_time is %d timesteps" % (run_time))
    print("Models will be saved every %d" % (save_time))

    ### END estimate of the ACTs so of the run_time and the 'distances' times! ###
    
    # Initial choice of Bij (interactions) based on the input_matrix
    B0           = B0
    for i in range(nparticles):
        for j in range(i+1,nparticles):            
            
            L      = 1        # Number of pairs of monomers that can potentially interact across the two segments
            lmbd   = abs(i-j) # The lenght of the chain that separates the two segments
            lmbd32 = 1. / (lmbd**alpha)

            
            # Nearest neighbors are not used to constrain the models
            if lmbd <= 1:
                interactions[i][j] = 100000.
                interactions[j][i] = 100000.
                continue



            #if input_matrix[i][j] == 1.:
            #    interactions[i][j] = 100.
            #    interactions[j][i] = 100.
            #    continue

            #if input_matrix[i][j] == 0.:
            #    interactions[i][j] = 0.
            #    interactions[j][i] = 0.
            #    continue

            interaction = B0 * lmbd32
            #interaction = B0 * log(1/(lmbd**1.5) * input_matrix[i][j]+1)
            interactions[i][j] = interaction
            interactions[j][i] = interaction
            print(i,j,lmbd,input_matrix[i][j],interaction)
                  
            #try:
            #interactions[i][j] = abs(B0*log(input_matrix[i][j]+1))
            #interactions[j][i] = abs(B0*log(input_matrix[i][j]+1))
            #print(i,j,input_matrix[i][j],input_matrix[i][j]+1,interactions[i][j])
            #except:
            #interactions[i][j] = 0.
            #interactions[i][j] = 0.
            #interactions[i][j] = B0
            #interactions[j][i] = B0
    #exit(1)
    fp_output = open("interactions_matrix.txt","w")
    for i in range(nparticles):
        for j in range(nparticles):
            fp_output.write("%d %d %f\n" % (i,j,interactions[i][j]))
    fp_output.close()
    #exit(1)
    tollerance = 10.0
    Dchi2    = 10
    nround = 0
    
    while Dchi2 <= tollerance:

        partition = {}
        radii     = {}
        interactions_parameters = {}
        for i in range(1,nparticles+1):
            partition[i] = i
            radii[i]     = 0.5
        #print(partition)
        #print(radii)
        for i in range(1,nparticles+1):
            for j in range(i+1,nparticles+1):
                pair = (i,j)                
                #print(i-1,j-1,interactions[i-1][j-1])
                interactions_parameters[pair] = ['attraction',interactions[i-1][j-1]] 
        compartmentalization = {
            'partition'    : partition,
            'radii'        : radii,
            'interactions' : interactions_parameters,
            'runtime'      : run_time,
            'distances'    : save_time
        }
        #print(compartmentalization)
        
        # Produce the models' population
        replica_seeds = []        
        for replica in range(nreplicas):
            replica_seeds.append(replica+1+initial_seed)
            #print(replica_seeds)
        pool = multiprocessing.Pool(processes=n_cpus) #, maxtasksperchild=n_cpus)

        results = []
        def collect_result(result):
            results.append((result[0], result[1], result[2]))

        start_time = time.time()            
        jobs = {}
        for replica_id, replica in enumerate(replica_seeds):
            replica_folder = lammps_folder + 'replica_' + str(replica) + '/'
            ini_conf  = "%s/replica_%s/Initial_conformation_replica_%s.txt" % (lammps_folder,replica,replica)
            tethering=False
            minimize=True
            compress_with_pbc=None
            compress_without_pbc=None
            initial_relaxation=None
            restrained_dynamics=None
            confining_environment=None
            steering_pairs=None
            time_dependent_steering_pairs=None
            loop_extrusion_dynamics_OLD=None
            loop_extrusion_dynamics=None            
            pbc=False
            keep_restart_out_dir2=None
            restart_file=False
            model_path=False
            store_n_steps=None
            useColvars=False
            chromosome_particle_numbers = [nparticles]
            timestep=timestep
            reset_timestep=0
            to_dump= save_time
            
            #print(replica_folder)
            jobs[replica] = partial(abortable_worker, run_lammps, timeout=3600*24*30,
                                    failedSeedLog=["runLog.txt", replica])
            #print(jobs)
            pool.apply_async(jobs[replica],
                             args=(replica, replica_folder, run_time,
                                   ini_conf, connectivity,
                                   neighbor,
                                   tethering, minimize,
                                   compress_with_pbc, compress_without_pbc,
                                   initial_relaxation,
                                   restrained_dynamics,
                                   confining_environment,
                                   steering_pairs,
                                   time_dependent_steering_pairs,
                                   compartmentalization,
                                   loop_extrusion_dynamics,
                                   to_dump, pbc, hide_log,
                                   gamma,timestep,
                                   reset_timestep,
                                   chromosome_particle_numbers,
                                   keep_restart_out_dir2,
                                   restart_file,
                                   model_path,
                                   store_n_steps,
                                   useColvars,), callback=collect_result)

        pool.close()
        pool.join()
        print("--- %.2f seconds --- generating models" % (time.time() - start_time))
        
        # Compute contact map from distances file
        start_time = time.time()            
        Nrec     = 0
        # Just counting Nrec
        for replica in range(1,nreplicas+1):
            for timestep_distances in range(compartmentalization['distances'],compartmentalization['runtime']+1,compartmentalization['distances']):                
                Nrec += 1

        contacts         = zeros((nparticles,nparticles), dtype = 'float')
        all_contacts     = zeros((Nrec,nparticles,nparticles), dtype = 'float')
        all_distances    = zeros((Nrec,nparticles,nparticles), dtype = 'float')
        E                = zeros(Nrec, dtype = 'float')
        exponent         = zeros(Nrec, dtype = 'float')
        all_energies     = zeros((Nrec,nparticles,nparticles), dtype = 'float')

        Nconf = 0
        for replica in range(1,nreplicas+1):
            for timestep_distances in range(compartmentalization['distances'],compartmentalization['runtime']+1,compartmentalization['distances']):                
                print("Analysing Nrec %s. Replica %s timestep %s" % (Nconf,replica,timestep_distances))
                all_contacts[Nconf]  = contacts_calculation(lammps_folder,replica,timestep_distances,nparticles,rc)
                all_distances[Nconf] = distances_calculation(lammps_folder,replica,timestep_distances,nparticles)
                E[Nconf],all_energies[Nconf] = energy_calculation(interactions,all_distances[Nconf])
                Nconf               += 1                
        print("--- %.2f seconds --- loading data" % (time.time() - start_time))
        
        print("--- Nrec %s ---" % Nrec)
        fp_output = open("all_contacts_round_%s_after_resampling.txt" % (nround),"w")
        for i in range(Nrec):
            for j in range(nparticles):
                for k in range(nparticles):
                    if all_contacts[i][j][k] != 0.0:
                        fp_output.write("%d %d %d %f\n" % (i,j,k,all_contacts[i][j][k]))
        fp_output.close()
        
        contacts = all_contacts.sum(axis=0)
        fp_output = open("contacts_round_%s_after_resampling_and_sum.txt" % (nround),"w")
        for i in range(nparticles):
            for j in range(nparticles):
                if i in masked or k in masked:
                    contacts[i][j] = 0
                if abs(i-j) <= 1:
                    contacts[i][j] = 0
                fp_output.write("%d %d %f\n" % (i,j,contacts[i][j]))
        fp_output.close()

        contacts = contacts / Nrec       
        fp_output = open("contacts_round_%s_after_resampling_and_sum_and_Nrec_division.txt" % (nround),"w")
        for i in range(nparticles):
            for j in range(nparticles):
                
                fp_output.write("%d %d %f\n" % (i,j,contacts[i][j]))
        fp_output.close()
        
        # Compute Chi2
        chi2 = chi2_giorgetti(contacts,input_matrix,sigma)
        print("At round",nround," chi 2",chi2)
        T = 1.0
        
        # The number of iterations is 10 times the number of particles!
        #Niterations = int(10*nparticles*(nparticles-1)*0.5)
        Niterations = 5000
        # Change 1 Bij at a time randomly
        for iterations in range(Niterations):

            # Change 1 Bij and re-compute the energy            
            start_time = time.time()            
            tmp_all_energies = np.copy(all_energies) 
            tmp_E            = np.copy(E)                
            tmp_interactions = np.copy(interactions)                
            tmp_contacts     = zeros((nparticles,nparticles), dtype = 'float')
            tmp_exponent     = np.copy(exponent)
            tmp_all_contacts = np.copy(all_contacts)
            print("--- %.2f seconds --- creating temporary objects" % (time.time() - start_time))

            # 1 - Extract pair to randomise
            start_time = time.time()            
            p1 = randint(0,nparticles-1)
            p2 = randint(0,nparticles-1)
            #while (all_distances[:,p1,p2].sum() == 10.*Nrec) or (p1 in masked or p2 in masked) or (abs(p1-p2) <= 1):
            while (p1 in masked or p2 in masked) or (abs(p1-p2) <= 1):
                p1 = randint(0,nparticles-1)
                p2 = randint(0,nparticles-1)

            add_Bij = uniform(-0.5,0.5)

            # 2 - Update tmp_interactions
            tmp_interactions[p1][p2] += add_Bij
            tmp_interactions[p2][p1] += add_Bij
            if tmp_interactions[p1][p2] < 0.0:
                tmp_interactions[p1][p2] = 0.0
                tmp_interactions[p2][p1] = 0.0
            #print(nround,iterations,p1,p2,interactions[p1][p2],interactions[p2][p1],add_Bij,tmp_interactions[p1][p2],tmp_interactions[p2][p1],array_equal(interactions,tmp_interactions))
            print("--- %.2f seconds --- randomising one Bij" % (time.time() - start_time))

            # 3 - Reweigh the contacts by the Boltzmann factor
            start_time = time.time()            
            for Nconf in range(Nrec):
                if iterations == 0 or all_distances[Nconf][p1][p2] <= 2.5:
                    tmp_all_energies[Nconf][p1][p2] = energy_calculation_single_element(tmp_interactions[p1][p2],all_distances[Nconf][p1][p2])
                    tmp_all_energies[Nconf][p2][p1] = tmp_all_energies[Nconf][p1][p2] 
                    tmp_E[Nconf]            = tmp_all_energies[Nconf].sum()
                    tmp_exponent[Nconf]     = exp(-(tmp_E[Nconf]-E[Nconf])/T)
                    tmp_all_contacts[Nconf] = all_contacts[Nconf] * tmp_exponent[Nconf]

            #for Nconf in range(Nrec):
            #    print(nround,Nrec,"tmp_E",tmp_E[Nconf],"E",E[Nconf],"tmp_exponent",tmp_exponent[Nconf],all_distances[Nconf][p1][p2])
                
            tmp_contacts = tmp_all_contacts.sum(axis=0) / tmp_exponent.sum()
            for i in range(nparticles):
                try:
                    tmp_contacts[i][i+1] = 0.0                        
                    tmp_contacts[i+1][i] = 0.0
                except:
                    pass
                tmp_contacts[i][i] = 0.0
                if i in masked or j in masked:
                    tmp_contacts[i][j] = 0.0
            print("--- %.2f seconds --- computing the weighted contact map" % (time.time() - start_time))
            
            # 4 - Compute the possible new chi2
            tmp_chi2 = chi2_giorgetti(tmp_contacts,input_matrix,sigma)
           
            start_time = time.time()            
            # 5 - If the possible new chi2 decreases, the new condition is stored
            if tmp_chi2 < chi2:
                print("Updating interactions, contacts and chi2 : chi2_Old = ",chi2," chi2_new = ",tmp_chi2)
                print("At iteration",iterations,", changed interaction",p1,p2,interactions[p1][p2],add_Bij,tmp_interactions[p1][p2])
                #print("Check changed quantities")
                #for Nconf in range(Nrec):
                #    print("Energy","New:",tmp_E[Nconf],"Old:",E[Nconf],"Exponent","New:",tmp_exponent[Nconf],"Old",exponent[Nconf],"Contacts","New:",tmp_all_contacts[Nconf][p1][p2],"Old:",all_contacts[Nconf][p1][p2])

                # Interactions, energies and chi2 are updated
                interactions = np.copy(tmp_interactions)
                E            = np.copy(tmp_E)
                all_energies = np.copy(tmp_all_energies)
                exponent     = np.copy(tmp_exponent)
                contacts     = np.copy(tmp_contacts)
                chi2         = tmp_chi2

                # Write the new interactions
                fp_output_interactions = open("interactions_round_%s_iteration_%s.txt" % (nround,iterations),"w")
                for i in range(nparticles):
                    for j in range(nparticles):
                        fp_output_interactions.write("%d %d %f\n" % (i,j,interactions[i][j]))
                fp_output_interactions.close()

                fp_output = open("contacts_round_%s_iteration_%s.txt" % (nround,iterations),"w")
                for i in range(nparticles):
                    for j in range(nparticles):
                        fp_output.write("%d %d %f\n" % (i,j,tmp_contacts[i][j]))
                fp_output.close()
                # Plot correlation of input and contact matrix
                # Plot interaction maps
            else:
                print("Not updating interactions, contacts and chi2 : chi2_Old = ",chi2," chi2_new = ",tmp_chi2)
                #print("At iteration",iterations,", changed interaction",p1,p2,interactions[p1][p2],add_Bij,tmp_interactions[p1][p2])
                #print("Check changed quantities")
                #for Nconf in range(Nrec):
                #    print("Energy","New:",tmp_E[Nconf],"Old:",E[Nconf],"Exponent","New:",tmp_exponent[Nconf],"Old",exponent[Nconf],"Contacts","New:",tmp_all_contacts[Nconf][p1][p2],"Old:",all_contacts[Nconf][p1][p2])
            print("--- %.2f seconds --- computing chi2 and updating" % (time.time() - start_time))
            #if iterations % 100 == 0: 
            #    print("Nround",nround,"Iteration",iterations)
            #exit(1)            
            #if iterations == 1:
            #    exit(1)
        Dchi2 = 10.
        nround += 1
        #exit(1)

### Auxiliary functions ###

def estimate_Ree_ACT(lammps_folder,replica):

    replica_folder = lammps_folder + 'replica_' + str(replica) + '_estimate_of_ACTs/'
    filename = replica_folder + "Ree.txt"
    
    fp = open(filename, "r")
    size = 0
    for line in fp.readlines():
        line = line.strip().split()
        if "#" in line:
            continue
        size += 1
        if size == 1:
            t1 = int(line[0])
        if size == 2:
            t2 = int(line[0])          
            deltat=t2-t1
        
    autocorr = zeros((size), dtype = 'float')
    counter  = zeros((size), dtype = 'float')
    Ree      = zeros((size,3), dtype = 'float')
    Threshold_integration = 0.01

    
    replica_folder = lammps_folder + 'replica_' + str(replica) + '_estimate_of_ACTs/'
    filename = replica_folder + "Ree.txt"
    
    fp = open(filename, "r")
    nl = 0
    for line in fp.readlines():
        line = line.strip().split()
        if "#" in line:
            continue
        nl += 1
        Ree[nl-1][0] = float(line[1])
        Ree[nl-1][1] = float(line[2])
        Ree[nl-1][2] = float(line[3])

    print("# Read %3d data points from file %s" % (nl-1,filename))
    print("# deltat = %d" % deltat)


    lag = 0
    for t in range(0,nl):
        autocorr[lag] += np.dot(Ree[t],Ree[t+lag])
        counter[lag]  += 1
        #print(nl)
    for lag in range(1,nl):
        for t in range(0,nl-lag,lag):
            autocorr[lag] += np.dot(Ree[t],Ree[t+lag])
            counter[lag]  += 1

    norm=autocorr[0]/counter[0]
    tau=0.0
    stop_integration=0
    fp = open("%sautocorrelation_Ree_vect.txt" % (replica_folder),"w")

    for t in range (0,nl):
        autocorr[t]=autocorr[t]/(counter[t]*norm)
        if autocorr[t]  < Threshold_integration: stop_integration=1
        if (autocorr[t] > Threshold_integration) and (stop_integration==0): tau+=autocorr[t]; 
        fp.write("%8.5lf %8.5lf\n" % (1.0*deltat*t,autocorr[t]))

    tau=tau*deltat;

    return tau

####

def estimate_Rg_ACT(lammps_folder,replica):

    replica_folder = lammps_folder + 'replica_' + str(replica) + '_estimate_of_ACTs/'
    filename = replica_folder + "Ree.txt"
    
    fp = open(filename, "r")
    size = 0
    for line in fp.readlines():
        line = line.strip().split()
        if "#" in line:
            continue
        size += 1
        if size == 1:
            t1 = int(line[0])
        if size == 2:
            t2 = int(line[0])          
            deltat=t2-t1

    autocorr = zeros((size), dtype = 'float')
    counter  = zeros((size), dtype = 'float')
    Rg       = zeros((size), dtype = 'float')
    Threshold_integration = 0.01

    
    replica_folder = lammps_folder + 'replica_' + str(replica) + '_estimate_of_ACTs/'
    filename = replica_folder + "Rg.txt"
    
    fp = open(filename, "r")
    nl = 0
    for line in fp.readlines():
        line = line.strip().split()
        if "#" in line:
            continue
        nl += 1
        if nl == 1:
            t1 = int(line[0])
        if nl == 2:
            t2 = int(line[0])          
            deltat=t2-t1
        Rg[nl-1] = float(line[1])

    print("# Read %3d data points from file %s" % (nl-1,filename))
    print("# deltat = %d" % deltat)

    lag = 0
    for t in range(0,nl):
        autocorr[lag] += Rg[t] * Rg[t+lag]
        counter[lag]  += 1
        #print(nl)
    for lag in range(1,nl):
        for t in range(0,nl-lag,lag):
            autocorr[lag] += Rg[t] * Rg[t+lag]
            counter[lag]  += 1

    norm=autocorr[0]/counter[0]
    tau=0.0
    stop_integration=0
    fp = open("%sautocorrelation_Rg_vect.txt" % (replica_folder),"w")

    for t in range (0,nl):
        autocorr[t]=autocorr[t]/(counter[t]*norm)
        if autocorr[t]  < Threshold_integration: stop_integration=1
        if (autocorr[t] > Threshold_integration) and (stop_integration==0): tau+=autocorr[t]; 
        fp.write("%8.5lf %8.5lf\n" % (1.0*deltat*t,autocorr[t]))

    tau=tau*deltat;

    return tau

####

def distances_calculation(lammps_folder,replica,timestep,nparticles):

    distances = ones((nparticles,nparticles), dtype = 'float') 
    distances = 10.*distances
    
    start_time = time.time()
    replica_folder = lammps_folder + 'replica_' + str(replica) + '/'
    fp_distances = open("%s/distances_%s.txt" % (replica_folder,timestep), "r")

    for line in fp_distances.readlines():
        line = line.strip().split()
        if line[0] == "#" or line[0] == "ITEM:" or len(line) < 3:
            continue            

        # Store the distances
        distances[int(line[0])-1][int(line[1])-1] = float(line[2])
        distances[int(line[1])-1][int(line[0])-1] = float(line[2])

    print("--- %.2f seconds --- distances calculation" % (time.time() - start_time))

    return distances

####

def contacts_calculation(lammps_folder,replica,timestep,nparticles,rc=1.5):

    contacts = zeros((nparticles,nparticles), dtype = 'float') 
    
    start_time = time.time()
    replica_folder = lammps_folder + 'replica_' + str(replica) + '/'
    fp_distances = open("%s/distances_%s.txt" % (replica_folder,timestep), "r")

    for line in fp_distances.readlines():
        line = line.strip().split()
        if line[0] == "#" or line[0] == "ITEM:" or len(line) < 3:
            continue            

        # Store the contacts                
        if float(line[2]) <= rc:
            contacts[int(line[0])-1][int(line[1])-1] += 1
            contacts[int(line[1])-1][int(line[0])-1] += 1

    print("Funct: contacts_calculation",contacts[3][1])
    print("--- %.2f seconds --- contacts calculation" % (time.time() - start_time))

    return contacts

####

#def energy_calculation(interactions,lammps_folder,replica,timestep):
def energy_calculation(interactions,distances):

    energy = 0

    start_time = time.time()
    
    sigma = 1.
    ratio = sigma / 2.5

    #print(distances)   
    contacts = distances < 1.5
    contacts = contacts.astype(int)
    #print(contacts)
    
    lj3 = 4.0  * interactions * pow(sigma,12.0)
    lj4 = 4.0  * interactions * pow(sigma,6.0)
            
    offset = 4.0 * interactions * (pow(ratio,12.0) - pow(ratio,6.0))
            
    rsq   = distances*distances
    with np.errstate(divide='ignore', invalid='ignore'):
        r2inv = 1.0/rsq
    r2inv[np.isnan(r2inv)] = 0
    r2inv[np.isinf(r2inv)] = 0    

    r6inv = r2inv*r2inv*r2inv
    
    evdwl = contacts * (r6inv*(lj3*r6inv-lj4) - offset)
    energy = evdwl.sum()

    #print(energy)
    #print("--- %.2f seconds --- energy calculation" % (time.time() - start_time))

    return energy, evdwl

####

def energy_calculation_single_element(interaction,distance):

    energy = 0

    start_time = time.time()
    
    sigma   = 1.0
    ratio   = sigma / 2.5

    #print(distances)
    if distance > 2.5 or distance == 0.0:
        energy = 0.0
        return energy
    
    contact = 1
    
    lj3 = 4.0  * interaction * pow(sigma,12.0)
    lj4 = 4.0  * interaction * pow(sigma,6.0)
            
    offset = 4.0 * interaction * (pow(ratio,12.0) - pow(ratio,6.0))
            
    rsq   = distance*distance
    r2inv = 1.0/rsq
    r6inv = r2inv*r2inv*r2inv
    
    energy = contact * (r6inv*(lj3*r6inv-lj4) - offset)

    #print(energy)
    #print("--- %.2f seconds --- energy calculation" % (time.time() - start_time))

    return energy

####

def chi2_giorgetti(contacts1, contacts2, sigma):

    #sigma : Estimate it as the experimental precision of contacts2, defined as the standard deviation between the counts in the duplicate experiments, normalized by Z
    n     = 1 #len(contacts1)

    start_time = time.time()
    chi2 = (contacts1 - contacts2)*(contacts1 - contacts2) / (sigma)
    chi2 = chi2.sum()
    print("--- %.2f seconds --- chi2_giorgetti" % (time.time() - start_time))

    return chi2

####

def read_trajectory_file(fname):

    coords=[]
    fhandler = open(fname)
    line = next(fhandler)
    try:
        while True:
            if line.startswith('ITEM: TIMESTEP'):
                while not line.startswith('ITEM: ATOMS'):
                    line = next(fhandler)
                if line.startswith('ITEM: ATOMS'):
                    line = next(fhandler)
            line = line.strip()
            if len(line) == 0:
                continue
            line_vals = line.split()
            coords += [float(line_vals[1]),float(line_vals[2]),float(line_vals[3])]
            line = next(fhandler)
    except StopIteration:
        pass
    fhandler.close()        
            
    return coords

def read_conformation_file(fname):

    mod={'x':[], 'y':[], 'z':[]}
    fhandler = open(fname)
    line = next(fhandler)
    try:
        while True:
            if line.startswith('LAMMPS input data file'):
                while not line.startswith(' Atoms'):
                    line = next(fhandler)
                if line.startswith(' Atoms'):
                    line = next(fhandler)
                    while len(line.strip()) == 0:
                        line = next(fhandler)
            line = line.strip()
            line_vals = line.split()
            mod['x'].append(float(line_vals[3]))
            mod['y'].append(float(line_vals[4]))
            mod['z'].append(float(line_vals[5]))
            line = next(fhandler)
            if len(line.strip()) == 0:
                break
    except StopIteration:
        pass
    fhandler.close()        
            
    return mod    

########## Part to perform the restrained dynamics ##########
# I should add here: The steered dynamics (Irene's and Hi-C based models) ; 
# The loop extrusion dynamics ; the binders based dynamics (Marenduzzo and Nicodemi)...etc...

def load_extruders(loop_extrusion_dynamics, extruders_positions, extruders_lifetimes, occupied_positions):

    loading_rate = loop_extrusion_dynamics['loading_rate']
    chrlengths   = loop_extrusion_dynamics['chrlength']
    resolution   = loop_extrusion_dynamics['resolution']
    lifetime     = loop_extrusion_dynamics['lifetime']

    #print(occupied_positions)
    print(loading_rate)


    offset = 0 
    for chrlength in chrlengths:
        #print("Offset",offset)
        for particle in range(1,chrlength+1):
            new_positions = [0,0]
            new_positions[0] = offset + particle
            new_positions[1] = offset + particle + 1
            # Check that the new proposed positions are not occupied
            if (new_positions[0] in occupied_positions) or (new_positions[1] in occupied_positions):
                continue
            if uniform(0,1) < loading_rate:
                #print("Position of the new extruder ",new_positions)
                #print("New extruder loaded at ",new_positions)
                extruders_positions.append(new_positions)
                # Initialise the lifetime of the new extruder
                extruders_lifetimes.append(int(0))
                sys.stdout.flush()
                #print("Extruders' positions (All %d)"      % (len(extruders_positions)),extruders_positions)        
                #print("Extruders' lifetimes (Variable %d)" % (len(extruders_lifetimes)),extruders_lifetimes)

        offset += chrlength
        
    #print("Extruders' positions (All %d)"      % (len(extruders_positions)),extruders_positions)        
    #print("Extruders' lifetimes (Variable %d)" % (len(extruders_lifetimes)),extruders_lifetimes)
    return(extruders_positions,extruders_lifetimes)

def unload_extruders(loop_extrusion_dynamics, extruders_positions, extruders_lifetimes, occupied_positions):

    #print(extruders_positions)
    unloading_rate = loop_extrusion_dynamics['unloading_rate'] #/ len(extruders_positions)
    chrlengths     = loop_extrusion_dynamics['chrlength']
    resolution     = loop_extrusion_dynamics['resolution']
    lifetime       = loop_extrusion_dynamics['lifetime']
    print(unloading_rate)

    _extruders_positions = [] #XXX
    _extruders_lifetimes = [] #XXX

    offset = 0 
    for extruder in range(len(extruders_positions)):
        if uniform(0,1) < unloading_rate: 
            print("Removing extruder due to unloading",extruder,extruders_positions[extruder],"with lifetime",extruders_lifetimes[extruder])
        else:
            _extruders_lifetimes.append(extruders_lifetimes[extruder])
            _extruders_positions.append(extruders_positions[extruder])  

    extruders_positions = _extruders_positions.copy()
    extruders_lifetimes = _extruders_lifetimes.copy()        
    #print("Extruders' positions (All %d)"      % (len(extruders_positions)),extruders_positions)        
    #print("Extruders' lifetimes (Variable %d)" % (len(extruders_lifetimes)),extruders_lifetimes)
    return(extruders_positions,extruders_lifetimes)
        

def linecount(filename):
    """
    Count valid lines of input colvars file
    
    :param filename: input colvars file.
    
    :returns: number of valid contact lines

    """
    
    k = 0
    tfp = open(filename)
    for i, line in enumerate(tfp):   

        if line.startswith('#'):
            continue
        cols_vals = line.split()
        if cols_vals[1] == cols_vals[2]:
            continue
        k += 1
        
    return k

##########

def generate_colvars_list(steering_pairs,
                          kincrease=0,
                          colvars_header='# collective variable: monitor distances\n\ncolvarsTrajFrequency 1000 # output every 1000 steps\ncolvarsRestartFrequency 10000000\n',
                          colvars_template='''

colvar {
  name %s
  # %s %s %i
  distance {
      group1 {
        atomNumbers %i
      }
      group2 {
        atomNumbers %i
      }
  }
}''',
                            colvars_tail = '''

harmonic {
  name h_pot_%s
  colvars %s
  centers %s
  forceConstant %f 
}\n''',                     colvars_harmonic_lower_bound_tail = '''

harmonicWalls {
  name hlb_pot_%s
  colvars %s
  lowerWalls %s
  forceConstant %f 
  lowerWallConstant 1.0
}\n'''
                            ):
                            
    """
    Generates lammps colvars file http://lammps.sandia.gov/doc/PDF/colvars-refman-lammps.pdf
    
    :param dict steering_pairs: dictionary containing all the information to write down the
      the input file for the colvars implementation
    :param exisiting_template colvars_header: header template for colvars file.
    :param exisiting_template colvars_template: contact template for colvars file.
    :param exisiting_template colvars_tail: tail template for colvars file.

    """

    # Getting the input
    # XXXThe target_pairs could be also a list as the one in output of get_HiCbased_restraintsXXX
    target_pairs                 = steering_pairs['colvar_input'] 
    outfile                      = steering_pairs['colvar_output'] 
    if 'kappa_vs_genomic_distance' in steering_pairs:
        kappa_vs_genomic_distance    = steering_pairs['kappa_vs_genomic_distance']
    if 'chrlength' in steering_pairs:
        chrlength                    = steering_pairs['chrlength']
    else:
        chrlength                    = 0
    if 'copies' in steering_pairs:
        copies                       = steering_pairs['copies']
    else:
        copies                       = ['A']
    kbin                         = 10000000
    binsize                      = steering_pairs['binsize']
    if 'percentage_enforced_contacts' in steering_pairs:
        percentage_enforced_contacts = steering_pairs['perc_enfor_contacts']
    else:
        percentage_enforced_contacts = 100

    # Here we extract from all the restraints only 
    # a random sub-sample of percentage_enforced_contacts/100*totcolvars
    rand_lines = []
    i=0
    j=0
    if isinstance(target_pairs, str):    
        totcolvars = linecount(target_pairs)
        ncolvars = int(totcolvars*(float(percentage_enforced_contacts)/100))
        
        #print "Number of enforced contacts = %i over %i" % (ncolvars,totcolvars)
        rand_positions = sample(list(range(totcolvars)), ncolvars)
        rand_positions = sorted(rand_positions)
    
        tfp = open(target_pairs)
        with open(target_pairs) as f:
            for line in f:
                line = line.strip()
                if j >= ncolvars:
                    break
                if line.startswith('#'):
                    continue
             
                cols_vals = line.split()
                # Avoid to enforce restraints between the same bin
                if cols_vals[1] == cols_vals[2]:
                    continue
            
                if i == rand_positions[j]:
                    rand_lines.append(line)
                    j += 1
                i += 1
        tfp.close()
    elif isinstance(target_pairs, HiCBasedRestraints):
        
        rand_lines = target_pairs.get_hicbased_restraints()
        totcolvars = len(rand_lines)
        ncolvars = int(totcolvars*(float(percentage_enforced_contacts)/100))
        
        #print "Number of enforced contacts = %i over %i" % (ncolvars,totcolvars)
        rand_positions = sample(list(range(totcolvars)), ncolvars)
        rand_positions = sorted(rand_positions)
        
        
    else:
        print("Unknown target_pairs")
        return    
    
        
    
    #print rand_lines

    seqdists = {}
    poffset=0
    outf = open(outfile,'w')
    outf.write(colvars_header)
    for copy_nbr in copies:
        i = 1
        for line in rand_lines:
            if isinstance(target_pairs, str):   
                cols_vals = line.split()
            else:
                cols_vals = ['chr'] + line
                
            #print cols_vals
            
            if isinstance(target_pairs, HiCBasedRestraints) and cols_vals[3] != "Harmonic" and cols_vals[3] != "HarmonicLowerBound":
                continue
            
            part1_start = int(cols_vals[1])*binsize
            part1_end = (int(cols_vals[1])+1)*binsize
            #print part1_start, part1_end

            part2_start = int(cols_vals[2])*binsize
            part2_end = (int(cols_vals[2])+1)*binsize
            #print part2_start, part2_end

            name = str(i)+copy_nbr  
            seqdist = abs(part1_start-part2_start)
            #print seqdist

            region1 = cols_vals[0] + '_' + str(part1_start) + '_' + str(part1_end)
            region2 = cols_vals[0] + '_' + str(part2_start) + '_' + str(part2_end)

            particle1 = int(cols_vals[1]) + 1 + poffset
            particle2 = int(cols_vals[2]) + 1 + poffset

            seqdists[name] = seqdist

            outf.write(colvars_template % (name,region1,region2,seqdist,particle1,particle2))
            
            if isinstance(target_pairs, HiCBasedRestraints):
                # If the spring constant is zero we avoid to add the restraint!
                if cols_vals[4] == 0.0:
                    continue
                 
                centre                 = cols_vals[5]
                kappa                  = cols_vals[4]*steering_pairs['k_factor']
                 
                if cols_vals[3] == "Harmonic":
                    outf.write(colvars_tail % (name,name,centre,kappa))
         
                if cols_vals[3] == "HarmonicLowerBound":
                    outf.write(colvars_harmonic_lower_bound_tail % (name,name,centre,kappa)) 
            
            i += 1
        poffset += chrlength
            
    outf.flush()
    
    #===========================================================================
    # if isinstance(target_pairs, HiCBasedRestraints):
    #     for copy_nbr in copies:
    #         i = 1
    #         for line in rand_lines:
    #             cols_vals = line
    #                             
    #             if cols_vals[3] == 0.0:
    #                 continue
    #             
    #             name = str(i)+copy_nbr 
    #             
    #             centre                 = cols_vals[4]
    #             kappa                  = cols_vals[3]
    #             
    #             if cols_vals[2] == "Harmonic":
    #                 outf.write(colvars_tail % (name,name,centre,kappa))
    #     
    #             elif cols_vals[2] == "HarmonicLowerBound":
    #                 outf.write(colvars_harmonic_lower_bound_tail % (name,name,centre,kappa))
    #             
    #                  
    #             
    #             i += 1
    #         poffset += chrlength
    #             
    #     outf.flush()
    #===========================================================================
    
    if 'kappa_vs_genomic_distance' in steering_pairs:   
            
        kappa_values = {}
        with open(kappa_vs_genomic_distance) as kgd:
            for line in kgd:
                line_vals = line.split()
                kappa_values[int(line_vals[0])] = float(line_vals[1])
            
        for seqd in set(seqdists.values()):
            kappa = 0
            if seqd in kappa_values:
                kappa = kappa_values[seqd]*kincrease
            else:
                for kappa_key in sorted(kappa_values, key=int):
                    if int(kappa_key) > seqd:
                        break
                    kappa = kappa_values[kappa_key]*kincrease
            centres=''
            names=''
            for seq_name in seqdists:
                if seqdists[seq_name] == seqd:
                    centres += ' 1.0'
                    names += ' '+seq_name
          
            outf.write(colvars_tail % (str(seqd),names,centres,kappa))
                    
    outf.flush()
    
    outf.close()
        
    
def generate_bond_list(steering_pairs):
                            
    """
    Generates lammps bond commands
    
    :param dict steering_pairs: dictionary containing all the information to write down the
      the input file for the bonds
    """

    # Getting the input
    # The target_pairs could be also a list as the one in output of get_HiCbased_restraintsXXX
    target_pairs                 = steering_pairs['colvar_input'] 
    if 'kappa_vs_genomic_distance' in steering_pairs:
        kappa_vs_genomic_distance    = steering_pairs['kappa_vs_genomic_distance']
    if 'chrlength' in steering_pairs:
        chrlength                    = steering_pairs['chrlength']
    else:
        chrlength                    = 0
    if 'copies' in steering_pairs:
        copies                       = steering_pairs['copies']
    else:
        copies                       = ['A']
    kbin                         = 10000000
    binsize                      = steering_pairs['binsize']
    if 'percentage_enforced_contacts' in steering_pairs:
        percentage_enforced_contacts = steering_pairs['perc_enfor_contacts']
    else:
        percentage_enforced_contacts = 100

    # Here we extract from all the restraints only 
    # a random sub-sample of percentage_enforced_contacts/100*totcolvars
    rand_lines = []
    i=0
    j=0
    if isinstance(target_pairs, str):    
        totcolvars = linecount(target_pairs)
        ncolvars = int(totcolvars*(float(percentage_enforced_contacts)/100))
        
        #print "Number of enforced contacts = %i over %i" % (ncolvars,totcolvars)
        rand_positions = sample(list(range(totcolvars)), ncolvars)
        rand_positions = sorted(rand_positions)
    
        tfp = open(target_pairs)
        with open(target_pairs) as f:
            for line in f:
                line = line.strip()
                if j >= ncolvars:
                    break
                if line.startswith('#'):
                    continue
             
                cols_vals = line.split()
                # Avoid to enforce restraints between the same bin
                if cols_vals[1] == cols_vals[2]:
                    continue
            
                if i == rand_positions[j]:
                    rand_lines.append(line)
                    j += 1
                i += 1
        tfp.close()
    elif isinstance(target_pairs, HiCBasedRestraints):
        
        rand_lines = target_pairs.get_hicbased_restraints()
        totcolvars = len(rand_lines)
        ncolvars = int(totcolvars*(float(percentage_enforced_contacts)/100))
        
        #print "Number of enforced contacts = %i over %i" % (ncolvars,totcolvars)
        rand_positions = sample(list(range(totcolvars)), ncolvars)
        rand_positions = sorted(rand_positions)
        
        
    else:
        print("Unknown target_pairs")
        return    
    
        
    
    #print rand_lines

    seqdists = {}
    poffset=0
    outf = []  #### a list
    for copy_nbr in copies:
        i = 1
        for line in rand_lines:
            if isinstance(target_pairs, str):   
                cols_vals = line.split()
            else:
                cols_vals = ['chr'] + line
                
            #print cols_vals
            
            if isinstance(target_pairs, HiCBasedRestraints) and cols_vals[3] != "Harmonic" and cols_vals[3] != "HarmonicLowerBound":
                continue
            
            part1_start = int(cols_vals[1])*binsize
            part1_end = (int(cols_vals[1])+1)*binsize
            #print part1_start, part1_end

            part2_start = int(cols_vals[2])*binsize
            part2_end = (int(cols_vals[2])+1)*binsize
            #print part2_start, part2_end

            name = str(i)+copy_nbr  
            seqdist = abs(part1_start-part2_start)
            #print seqdist

            region1 = cols_vals[0] + '_' + str(part1_start) + '_' + str(part1_end)
            region2 = cols_vals[0] + '_' + str(part2_start) + '_' + str(part2_end)

            particle1 = int(cols_vals[1]) + 1 + poffset
            particle2 = int(cols_vals[2]) + 1 + poffset

            seqdists[name] = seqdist

            
            if isinstance(target_pairs, HiCBasedRestraints):
                # If the spring constant is zero we avoid to add the restraint!
                if cols_vals[4] == 0.0:
                    continue
                 
                centre                 = cols_vals[5]
                kappa                  = cols_vals[4]*steering_pairs['k_factor']
                 
                bondType = None
                if cols_vals[3] == "Harmonic":
                    bondType = 'bond'
                elif cols_vals[3] == "HarmonicLowerBound":
                    bondType = 'lbound'

                if bondType:
                    outf.append('fix %s all restrain %s %d %d %f %f %f %f' %(
                        name, bondType, particle1, particle2, 0, kappa, 
                        centre, centre))

            
            i += 1
        poffset += chrlength

    return outf

##########

def generate_time_dependent_bond_list(steering_pairs):


    """
    Generates lammps bond commands
    
    :param dict steering_pairs: dictionary containing all the information to write down the
      the input file for the bonds
    """

    outf = []  #### a list
    # Defining the particle pairs
    for pair in steering_pairs:

        #print steering_pairs[pair]
        sys.stdout.flush()
        for i in range(len(steering_pairs[pair][0])):
            name    = "%s_%s_%s" % (i, int(pair[0])+1, int(pair[1])+1)
            seqdist = abs(int(pair[1])-int(pair[0])) 
            particle1 = int(pair[0])+1
            particle2 = int(pair[1])+1

            restraint_type         = steering_pairs[pair][0][i]
            kappa_start            = steering_pairs[pair][1][i]
            kappa_stop             = steering_pairs[pair][2][i]
            centre_start           = steering_pairs[pair][3][i]
            centre_stop            = steering_pairs[pair][4][i]
            timesteps_per_k_change = steering_pairs[pair][5][i]     

            bonType = None
            if restraint_type == "Harmonic":
                bonType = 'bond'
            elif restraint_type == "HarmonicLowerBound":
                bonType = 'lbound'

            if bonType:
                outf.append('fix %s all restrain %s %d %d %f %f %f %f' %(
                    name, bonType, particle1, particle2, kappa_start, kappa_stop, 
                    centre_start, centre_stop))
    return outf  

##########

def generate_time_dependent_colvars_list(steering_pairs,
                                         outfile,
                                         colvar_dump_freq,
                                         colvars_header='# collective variable: monitor distances\n\ncolvarsTrajFrequency %i # output every %i steps\ncolvarsRestartFrequency 1000000\n',
                                         colvars_template='''

colvar {
  name %s
  # %s %s %i
  width 1.0
  distance {
      group1 {
        atomNumbers %i
      }
      group2 {
        atomNumbers %i
      }
  }
}''',
                                         colvars_harmonic_tail = '''

harmonic {
  name h_pot_%s
  colvars %s
  forceConstant %f       
  targetForceConstant %f 
  centers %s             
  targetCenters %s       
  targetNumSteps %s
  outputEnergy   yes
}\n''',
                                         colvars_harmonic_lower_bound_tail = '''
harmonicBound {
  name hlb_pot_%s
  colvars %s
  forceConstant %f
  targetForceConstant %f
  centers %f
  targetCenters %f
  targetNumSteps %s
  outputEnergy   yes
}\n'''
                            ):


    """
    harmonicWalls {
    name hlb_pot_%s
    colvars %s
    forceConstant %f       # This is the force constant at time_point
    targetForceConstant %f # This is the force constant at time_point+1
    centers %f             # This is the equilibrium distance at time_point+1
    targetCenters %f      # This is the equilibrium distance at time_point+1
    targetNumSteps %d      # This is the number of timesteps between time_point and time_point+1
    outputEnergy   yes
    }\n''',


    colvars_harmonic_lower_bound_tail = '''
    
    harmonicBound {
    name hlb_pot_%s
    colvars %s
    forceConstant %f       # This is the force constant at time_point
    targetForceConstant %f # This is the force constant at time_point+1
    centers %f             # This is the equilibrium distance at time_point+1
    targetCenters %f      # This is the equilibrium distance at time_point+1
    targetNumSteps %d      # This is the number of timesteps between time_point and time_point+1
    outputEnergy   yes
    }\n''',
    
    Generates lammps colvars file http://lammps.sandia.gov/doc/PDF/colvars-refman-lammps.pdf
    
    :param dict steering_pairs: dictionary containing all the information to write down the
      the input file for the colvars implementation
    :param exisiting_template colvars_header: header template for colvars file.
    :param exisiting_template colvars_template: contact template for colvars file.
    :param exisiting_template colvars_tail: tail template for colvars file.

    """

    #restraints[pair] = [time_dependent_restraints[time_point+1][pair][0],     # Restraint type -> Is the one at time point time_point+1
    #time_dependent_restraints[time_point][pair][1]*10.,                       # Initial spring constant 
    #time_dependent_restraints[time_point+1][pair][1]*10.,                     # Final spring constant 
    #time_dependent_restraints[time_point][pair][2],                           # Initial equilibrium distance 
    #time_dependent_restraints[time_point+1][pair][2],                         # Final equilibrium distance 
    #int(time_dependent_steering_pairs['timesteps_per_k_change'][time_point])] # Number of timesteps for the gradual change

    outf = open(outfile,'w')
    #tfreq=10000
    #for pair in steering_pairs:
    #    if len(steering_pairs[pair][5]) >= 1:
    #        tfreq = int(steering_pairs[pair][5][0]/100)
    #        break
    
    tfreq = colvar_dump_freq
    outf.write(colvars_header % (tfreq, tfreq))
    # Defining the particle pairs
    for pair in steering_pairs:

        #print steering_pairs[pair]
        sys.stdout.flush()
        for i in range(len(steering_pairs[pair][0])):
            name    = "%s_%s_%s" % (i, int(pair[0])+1, int(pair[1])+1)
            seqdist = abs(int(pair[1])-int(pair[0])) 
            region1 = "particle_%s" % (int(pair[0])+1)
            region2 = "particle_%s" % (int(pair[1])+1)
            
            outf.write(colvars_template % (name,region1,region2,seqdist,int(pair[0])+1,int(pair[1])+1))

            restraint_type         = steering_pairs[pair][0][i]
            kappa_start            = steering_pairs[pair][1][i]
            kappa_stop             = steering_pairs[pair][2][i]
            centre_start           = steering_pairs[pair][3][i]
            centre_stop            = steering_pairs[pair][4][i]
            timesteps_per_k_change = steering_pairs[pair][5][i]     

            if restraint_type == "Harmonic":
                outf.write(colvars_harmonic_tail             % (name,name,kappa_start,kappa_stop,centre_start,centre_stop,timesteps_per_k_change))
                
            if restraint_type == "HarmonicLowerBound":
                outf.write(colvars_harmonic_lower_bound_tail % (name,name,kappa_start,kappa_stop,centre_start,centre_stop,timesteps_per_k_change))



                    
    outf.flush()
    
    outf.close()  

##########

def get_time_dependent_colvars_list(time_dependent_steering_pairs):
                            
    """
    Generates lammps colvars file http://lammps.sandia.gov/doc/PDF/colvars-refman-lammps.pdf
    
    :param dict time_dependent_steering_pairs: dictionary containing all the information to write down the
      the input file for the colvars implementation
    """

    # Getting the input
    # XXXThe target_pairs_file could be also a list as the one in output of get_HiCbased_restraintsXXX
    target_pairs            = time_dependent_steering_pairs['colvar_input'] 
    outfile                      = time_dependent_steering_pairs['colvar_output']
    if 'chrlength' in time_dependent_steering_pairs:
        chrlength                    = time_dependent_steering_pairs['chrlength']
    binsize                      = time_dependent_steering_pairs['binsize']
    if 'percentage_enforced_contacts' in time_dependent_steering_pairs:
        percentage_enforced_contacts = time_dependent_steering_pairs['perc_enfor_contacts']
    else:
        percentage_enforced_contacts = 100

    # HiCbasedRestraints is a list of restraints returned by this function.
    # Each entry of the list is a list of 5 elements describing the details of the restraint:
    # 0 - particle_i
    # 1 - particle_j
    # 2 - type_of_restraint = Harmonic or HarmonicLowerBound or HarmonicUpperBound
    # 3 - the kforce of the restraint
    # 4 - the equilibrium (or maximum or minimum respectively) distance associated to the restraint
   
    # Here we extract from all the restraints only a random sub-sample
    # of percentage_enforced_contacts/100*totcolvars
    rand_lines = []
    i=0
    j=0
    if isinstance(target_pairs, str):    
        time_dependent_restraints = {}
        totcolvars = linecount(target_pairs)
        ncolvars = int(totcolvars*(float(percentage_enforced_contacts)/100))
        
        #print "Number of enforced contacts = %i over %i" % (ncolvars,totcolvars)
        rand_positions = sample(list(range(totcolvars)), ncolvars)
        rand_positions = sorted(rand_positions)
        
        with open(target_pairs) as f:
            for line in f:
                line = line.strip()
                if j >= ncolvars:
                    break
                if line.startswith('#') or line == "":
                    continue
                
                # Line format: timepoint,particle1,particle2,restraint_type,kforce,distance
                cols_vals = line.split()
                
                if int(cols_vals[1]) < int(cols_vals[2]):                
                    pair = (int(cols_vals[1]), int(cols_vals[2]))
                else:
                    pair = (int(cols_vals[2]), int(cols_vals[1]))
            
                try:
                    if pair    in time_dependent_restraints[int(cols_vals[0])]:
                        print("WARNING: Check your restraint list! pair %s is repeated in time point %s!" % (pair, int(cols_vals[0])))
                    # List content: restraint_type,kforce,distance
                    time_dependent_restraints[int(cols_vals[0])][pair] = [cols_vals[3],
                                                                          float(cols_vals[4]),
                                                                          float(cols_vals[5])]
                except:
                    time_dependent_restraints[int(cols_vals[0])] = {}
                    # List content: restraint_type,kforce,distance
                    time_dependent_restraints[int(cols_vals[0])][pair] = [cols_vals[3],
                                                                          float(cols_vals[4]),
                                                                          float(cols_vals[5])]
                if i == rand_positions[j]:
                    rand_lines.append(line)
                    j += 1
                i += 1
    elif isinstance(target_pairs, list):
        time_dependent_restraints = dict((i,{}) for i in range(len(target_pairs)))
        for time_point, HiCR in enumerate(target_pairs):
            rand_lines = HiCR.get_hicbased_restraints()
            totcolvars = len(rand_lines)
            ncolvars = int(totcolvars*(float(percentage_enforced_contacts)/100))
            
            #print "Number of enforced contacts = %i over %i" % (ncolvars,totcolvars)
            rand_positions = sample(list(range(totcolvars)), ncolvars)
            rand_positions = sorted(rand_positions)
            
            for cols_vals in rand_lines:
                
                if cols_vals[2] != "Harmonic" and cols_vals[2] != "HarmonicLowerBound":
                    continue
                if int(cols_vals[0]) < int(cols_vals[1]):                
                    pair = (int(cols_vals[0]), int(cols_vals[1]))
                else:
                    pair = (int(cols_vals[1]), int(cols_vals[0]))
            
                if pair in time_dependent_restraints[time_point]:
                    print("WARNING: Check your restraint list! pair %s is repeated in time point %s!" % (pair, time_point))
                # List content: restraint_type,kforce,distance
                time_dependent_restraints[time_point][pair] = [cols_vals[2],
                                                                      float(cols_vals[3]),
                                                                      float(cols_vals[4])]
        
    else:
        print("Unknown target_pairs")
        return 

#     for time_point in sorted(time_dependent_restraints.keys()):
#         for pair in time_dependent_restraints[time_point]:
#             print "#Time_dependent_restraints", time_point,pair, time_dependent_restraints[time_point][pair]
    return time_dependent_restraints



### TODO Add the option to add also spheres of different radii (e.g. to simulate nucleoli)
########## Part to generate the initial conformation ##########
def generate_chromosome_random_walks_conformation ( chromosome_particle_numbers ,
                                                    confining_environment = None ,
                                                    particle_radius=0.5 ,
                                                    seed_of_the_random_number_generator=1 ,
                                                    number_of_conformations=1,
                                                    outfile="Initial_random_walk_conformation.dat",
                                                    atom_types=None,
                                                    pbc=False,
                                                    center=True):
    """
    Generates lammps initial conformation file by random walks
    
    :param chromosome_particle_numbers: list with the number of particles of each chromosome.    
    :param ['sphere',100.] confining_environment: dictionary with the confining environment of the conformation
            Possible confining environments:
            ['cube',edge_width]
            ['sphere',radius]
            ['ellipsoid',x-semiaxes, y-semiaxes, z-semiaxes]
            ['cylinder', basal radius, height]
    :param 0.5 particle_radius: Radius of each particle.
    :param 1 seed_of_the_random_number_generator: random seed.
    :param 1 number_of_conformations: copies of the conformation.
    :param outfile: file where to store resulting initial conformation file

    """
    seed(seed_of_the_random_number_generator)
    
    # This allows to organize the largest chromosomes first.
    # This is to get a better acceptance of the chromosome positioning.
    chromosome_particle_numbers = [int(x) for x in chromosome_particle_numbers]
    chromosome_particle_numbers.sort(key=int,reverse=True)

    for cnt in range(number_of_conformations):

        final_random_walks = generate_random_walks(chromosome_particle_numbers,
                                                   particle_radius,
                                                   confining_environment,
                                                   pbc=pbc,
                                                   center=center)

        # Writing the final_random_walks conformation
        #print "Succesfully generated conformation number %d\n" % (cnt+1)
        write_initial_conformation_file(final_random_walks,
                                        chromosome_particle_numbers,
                                        confining_environment,
                                        atom_types=atom_types,
                                        out_file=outfile)

##########
        
def generate_chromosome_rosettes_conformation ( chromosome_particle_numbers ,
                                                fractional_radial_positions=None,
                                                confining_environment=['sphere',100.] ,
                                                rosette_radius=12.0 , particle_radius=0.5 ,
                                                seed_of_the_random_number_generator=1 ,
                                                number_of_conformations=1,
                                                outfile = "Initial_rosette_conformation.dat",
                                                atom_types=1):
    """
    Generates lammps initial conformation file by rosettes conformation
    
    :param chromosome_particle_numbers: list with the number of particles of each chromosome.    
    :param None fractional_radial_positions: list with fractional radial positions for all the chromosomes.
    :param ['sphere',100.] confining_environment: dictionary with the confining environment of the conformation
            Possible confining environments:
            ['cube',edge_width]
            ['sphere',radius]
            ['ellipsoid',x-semiaxes, y-semiaxes, z-semiaxes]
            ['cylinder', basal radius, height]
    :param 0.5 particle_radius: Radius of each particle.
    :param 1 seed_of_the_random_number_generator: random seed.
    :param 1 number_of_conformations: copies of the conformation.
    :param outfile: file where to store resulting initial conformation file

    """
    seed(seed_of_the_random_number_generator)
    
    # This allows to organize the largest chromosomes first.
    # This is to get a better acceptance of the chromosome positioning.
    chromosome_particle_numbers = [int(x) for x in chromosome_particle_numbers]
    chromosome_particle_numbers.sort(key=int,reverse=True)    

    initial_rosettes , rosettes_lengths = generate_rosettes(chromosome_particle_numbers,
                                                            rosette_radius,
                                                            particle_radius)
    print(rosettes_lengths)

    
    # Constructing the rosettes conformations
    for cnt in range(number_of_conformations):

        temptative = 0
        particle_inside   = 0 # 0 means a particle is outside
        particles_overlap = 0 # 0 means two particles are overlapping
        while particle_inside == 0 or particles_overlap == 0:
            temptative += 1
            print("Temptative number %d" % temptative)
            particle_inside   = 1
            particles_overlap = 1
            segments_P1 = []
            segments_P0 = []
            side = 0
            init_rosettes = copy.deepcopy(initial_rosettes)
            
            # Guess of the initial segment conformation:
            # 1 - each rod is placed inside the confining evironment
            # in a random position and with random orientation
            # 2 - possible clashes between generated rods are checked
            if fractional_radial_positions:
                if len(fractional_radial_positions) != len(chromosome_particle_numbers):
                    print("Please provide the desired fractional radial positions for all the chromosomes")
                    sys.exit()
                segments_P1 , segments_P0 = generate_rods_biased_conformation(rosettes_lengths, rosette_radius,
                                                                              confining_environment,
                                                                              fractional_radial_positions,
                                                                              max_number_of_temptative=1000)
            else:
                segments_P1 , segments_P0 = generate_rods_random_conformation(rosettes_lengths, rosette_radius,
                                                                              confining_environment,
                                                                              max_number_of_temptative=1000)

            # Roto-translation of the rosettes according to the segment position and orientation 
            final_rosettes = rosettes_rototranslation(init_rosettes, segments_P1, segments_P0)
            
            # Checking that the beads are all inside the confining environment and are not overlapping
            for r in range(len(final_rosettes)):
                molecule0 = list(zip(final_rosettes[r]['x'],final_rosettes[r]['y'],final_rosettes[r]['z']))
                print(len(molecule0),len(molecule0[0]))
                if particle_inside == 0:
                    break
                for i in range(len(molecule0)):
                    # Check if the particle is inside the confining_environment

                    particle_inside = check_point_inside_the_confining_environment(molecule0[i][0],
                                                                                   molecule0[i][1],
                                                                                   molecule0[i][2],
                                                                                   1.0,
                                                                                   confining_environment)
                    if particle_inside == 0:
                        break

            if particle_inside == 1:
                for rosette_pair in list(combinations(final_rosettes,2)):
                    molecule0 = list(zip(rosette_pair[0]['x'],rosette_pair[0]['y'],rosette_pair[0]['z']))
                    molecule1 = list(zip(rosette_pair[1]['x'],rosette_pair[1]['y'],rosette_pair[1]['z']))
                    distances = spatial.distance.cdist(molecule1,molecule0)
                    print("Different chromosomes",len(molecule0),len(molecule0[0]),distances.min())
                    if distances.min() < particle_radius*2.0*0.95:
                        particles_overlap = 0
                        break

                if particles_overlap != 0:
                    for r in range(len(final_rosettes)):
                        molecule0 = list(zip(final_rosettes[r]['x'],final_rosettes[r]['y'],final_rosettes[r]['z']))
                        print(len(molecule0),len(molecule0[0]))

                        distances = spatial.distance.cdist(molecule0,molecule0)
                        print("Same chromosome",distances.min())
                        sys.stdout.flush()
                        for i in range(len(molecule0)):
                            for j in range(i+1,len(molecule0)):
                                if distances[(i,j)] < particle_radius*2.0*0.95:
                                    particles_overlap = 0
                                    print("Particles",i,"and",j,"are contacting",distances[(i,j)])
                                    sys.stdout.flush()
                                if particles_overlap == 0:
                                    break
                            if particles_overlap == 0:
                                break
                        if particles_overlap == 0:
                            break
            
        # Writing the final_rosettes conformation
        print("Succesfully generated conformation number %d\n" % (cnt+1))
        write_initial_conformation_file(final_rosettes,
                                        chromosome_particle_numbers,
                                        confining_environment,
                                        out_file=outfile,
                                        atom_types=atom_types)

##########
        
def generate_chromosome_rosettes_conformation_with_pbc ( chromosome_particle_numbers ,
                                                         fractional_radial_positions=None,
                                                         confining_environment=['cube',100.] ,
                                                         rosette_radius=12.0 , particle_radius=0.5 ,
                                                         seed_of_the_random_number_generator=1 ,
                                                         number_of_conformations=1,
                                                         outfile = "Initial_rosette_conformation_with_pbc.dat",
                                                         atom_types=1, k=6., x=0.38, p=1.0):
    """
    Generates lammps initial conformation file by rosettes conformation
    
    :param chromosome_particle_numbers: list with the number of particles of each chromosome.    
    :param None fractional_radial_positions: list with fractional radial positions for all the chromosomes.
    :param ['cube',100.] confining_environment: dictionary with the confining environment of the conformation
            Possible confining environments:
            ['cube',edge_width]
    :param 0.5 particle_radius: Radius of each particle.
    :param 1 seed_of_the_random_number_generator: random seed.
    :param 1 number_of_conformations: copies of the conformation.
    :param outfile: file where to store resulting initial conformation file

    """
    seed(seed_of_the_random_number_generator)
    
    # This allows to organize the largest chromosomes first.
    # This is to get a better acceptance of the chromosome positioning.
    chromosome_particle_numbers = [int(x) for x in chromosome_particle_numbers]
    chromosome_particle_numbers.sort(key=int,reverse=True)    

    initial_rosettes , rosettes_lengths = generate_rosettes(chromosome_particle_numbers,
                                                            rosette_radius,
                                                            particle_radius,
                                                            k=k, x=x, p=p)
    print(rosettes_lengths)

    
    # Constructing the rosettes conformations
    for cnt in range(number_of_conformations):

        particles_overlap = 0 # 0 means two particles are overlapping
        while particles_overlap == 0:
            particles_overlap = 1
            segments_P1 = []
            segments_P0 = []
            side = 0
            init_rosettes = copy.deepcopy(initial_rosettes)
            
            # Guess of the initial segment conformation:
            # 1 - each rod is placed in a random position and with random orientation
            # 2 - possible clashes between generated rods are checked taking into account pbc
            segments_P1 , segments_P0 = generate_rods_random_conformation_with_pbc (
                rosettes_lengths, 
                rosette_radius,
                confining_environment,
                max_number_of_temptative=100000)

            # Roto-translation of the rosettes according to the segment position and orientation 
            final_rosettes = rosettes_rototranslation(init_rosettes, segments_P1, segments_P0)
            
            # Checking that the beads once folded inside the simulation box (for pbc) are not overlapping
            folded_rosettes = copy.deepcopy(final_rosettes)
            for r in range(len(folded_rosettes)):
                particle = 0
                for x, y, z in zip(folded_rosettes[r]['x'],folded_rosettes[r]['y'],folded_rosettes[r]['z']):
                    #inside_1 = check_point_inside_the_confining_environment(x, y, z,
                    #                                                        particle_radius,
                    #                                                        confining_environment)
                    #if inside_1 == 0:
                    #    print inside_1, r, particle, x, y, z
                    
                    while x >  (confining_environment[1]*0.5):
                        x -=  confining_environment[1]
                    while x < -(confining_environment[1]*0.5):
                        x +=  confining_environment[1]
                            
                    while y >  (confining_environment[1]*0.5):
                        y -=  confining_environment[1]
                    while y < -(confining_environment[1]*0.5):
                        y +=  confining_environment[1]
                                    
                    while z >  (confining_environment[1]*0.5):
                        z -=  confining_environment[1]
                    while z < -(confining_environment[1]*0.5):
                        z +=  confining_environment[1]

                    #inside_2 = check_point_inside_the_confining_environment(x, y, z,
                    #                                                      particle_radius,
                    #                                                      confining_environment)
                    #if inside_2 == 1 and inside_1 == 0:
                    #    print inside_2, r, particle, x, y, z
                    folded_rosettes[r]['x'][particle] = x
                    folded_rosettes[r]['y'][particle] = y
                    folded_rosettes[r]['z'][particle] = z
                    particle += 1

            print(len(folded_rosettes))
            sys.stdout.flush()
            if len(folded_rosettes) > 1:
                for rosette_pair in list(combinations(folded_rosettes,2)):
                    molecule0 = list(zip(rosette_pair[0]['x'],rosette_pair[0]['y'],rosette_pair[0]['z']))
                    molecule1 = list(zip(rosette_pair[1]['x'],rosette_pair[1]['y'],rosette_pair[1]['z']))
                    distances = spatial.distance.cdist(molecule1,molecule0)
                    print(len(molecule0),len(molecule0[0]),"Minimum distance",distances.min())
                    if distances.min() < particle_radius*2.0*0.95:
                        particles_overlap = 0
                        break

            if particles_overlap != 0:
                for r in range(len(folded_rosettes)):
                    molecule0 = list(zip(folded_rosettes[r]['x'],folded_rosettes[r]['y'],folded_rosettes[r]['z']))
                    print(len(molecule0),len(molecule0[0]))

                    distances = spatial.distance.cdist(molecule0,molecule0)
                    print("Minimum distance", distances.min())
                    for i in range(len(molecule0)):
                        for j in range(i+1,len(molecule0)):
                            if distances[(i,j)] < particle_radius*2.0*0.7:
                                particles_overlap = 0
                                print("Particles",i,"and",j,"are contacting",distances[(i,j)])
                                sys.stdout.flush()
                            if particles_overlap == 0:
                                break
                        if particles_overlap == 0:
                            break
                    if particles_overlap == 0:
                        break

            
        # Writing the final_rosettes conformation
        print("Succesfully generated conformation number %d\n" % (cnt+1))
        write_initial_conformation_file(final_rosettes,
                                        chromosome_particle_numbers,
                                        confining_environment,
                                        out_file=outfile,
                                        atom_types=atom_types)

##########

def generate_rosettes(chromosome_particle_numbers, rosette_radius, particle_radius, k=6., x=0.38, p=1.0):
    # Genaration of the rosettes
    # XXXA. Rosa publicationXXX
    # List to contain the rosettes and the rosettes lengths
    rosettes = []
    rosettes_lengths  = []

    for number_of_particles in chromosome_particle_numbers:
        
        # Variable to build the chain
        phi = 0.0

        # Dictory of lists to contain the rosette
        rosette      = {}
        rosette['x'] = []
        rosette['y'] = []
        rosette['z'] = []        
        
        # Position of the first particle (x_0, 0.0, 0.0)
        rosette['x'].append(rosette_radius * (x + (1 - x) * cos(k*phi) * cos(k*phi)) * cos(phi))
        rosette['y'].append(rosette_radius * (x + (1 - x) * cos(k*phi) * cos(k*phi)) * sin(phi))
        rosette['z'].append(2.0 * particle_radius * phi / (2.0 * pi))
        #print "First bead is in position: %f %f %f" % (rosette['x'][0], rosette['y'][0], rosette['z'][0])

        # Building the chain: The rosette is growing along the positive z-axes
        for particle in range(1,number_of_particles):

            distance = 0.0
            while distance < (particle_radius*2.0): 
                phi   = phi + 0.001
                x_tmp = rosette_radius * (x + (1 - x) * cos(k*phi) * cos(k*phi)) * cos(phi)
                y_tmp = rosette_radius * (x + (1 - x) * cos(k*phi) * cos(k*phi)) * sin(phi)
                z_tmp = 2.0 * particle_radius * phi / (2.0 * pi)     
                distance  = sqrt((x_tmp - rosette['x'][-1])*(x_tmp - rosette['x'][-1]) +
                                 (y_tmp - rosette['y'][-1])*(y_tmp - rosette['y'][-1]) +
                                 (z_tmp - rosette['z'][-1])*(z_tmp - rosette['z'][-1]))

            rosette['x'].append(x_tmp)
            rosette['y'].append(y_tmp)
            rosette['z'].append(z_tmp)
            if distance > ((particle_radius*2.0)*1.2):
                print("%f %d %d %d" % (distance, particle-1, particle))
            
        rosettes.append(rosette)
        rosettes_lengths.append(rosette['z'][-1]-rosette['z'][0])
        
    return rosettes , rosettes_lengths

##########

def generate_rods_biased_conformation(rosettes_lengths, rosette_radius,
                                      confining_environment,
                                      fractional_radial_positions,
                                      max_number_of_temptative=100000):
    # Construction of the rods initial conformation 
    segments_P0 = []
    segments_P1 = []

    if confining_environment[0] != 'sphere':
        print("ERROR: Biased chromosome positioning is currently implemented")
        print("only for spherical confinement. If you need other shapes, please")
        print("contact the developers")
    
    for length , target_radial_position in zip(rosettes_lengths,fractional_radial_positions):
        tentative            = 0
        clashes              = 0 # 0 means that there is an clash -> PROBLEM
        best_radial_position = 1.0
        best_radial_distance = 1.0
        best_segment_P0      = []
        best_segment_P1      = []
        
        # Positioning the rods
        while tentative < 100000 and best_radial_distance > 0.00005:                

            print("Length = %f" % length)

            print("Trying to position terminus 0")
            segment_P0_tmp = []
            segment_P0_tmp = draw_point_inside_the_confining_environment(confining_environment,
                                                                         rosette_radius,
                                                                         length)
            print("Successfully positioned terminus 0: %f %f %f" % (segment_P0_tmp[0], segment_P0_tmp[1], segment_P0_tmp[2]))
            
            print("Trying to position terminus 1")
            segment_P1_tmp = []                            
            segment_P1_tmp = draw_second_extreme_of_a_segment_inside_the_confining_environment(segment_P0_tmp[0],
                                                                                               segment_P0_tmp[1],
                                                                                               segment_P0_tmp[2],
                                                                                               length,
                                                                                               rosette_radius,
                                                                                               confining_environment)
            print("Successfully positioned terminus 1: %f %f %f" % (segment_P1_tmp[0], segment_P1_tmp[1], segment_P1_tmp[2]))

            # Check clashes with the previously positioned rods
            clashes = 1
            for segment_P1,segment_P0 in zip(segments_P1,segments_P0):
                clashes = check_segments_clashes(segment_P1,
                                                 segment_P0,
                                                 segment_P1_tmp,
                                                 segment_P0_tmp,
                                                 rosette_radius)
                if clashes == 0:
                    break                

            if clashes == 1:
                # Check whether the midpoint of the segment is close to the target radial position
                segment_midpoint = []
                segment_midpoint.append((segment_P0_tmp[0] + segment_P1_tmp[0])*0.5)
                segment_midpoint.append((segment_P0_tmp[1] + segment_P1_tmp[1])*0.5)
                segment_midpoint.append((segment_P0_tmp[2] + segment_P1_tmp[2])*0.5)

                radial_position = sqrt( ( segment_midpoint[0] * segment_midpoint[0] +
                                          segment_midpoint[1] * segment_midpoint[1] +
                                          segment_midpoint[2] * segment_midpoint[2] ) /
                                        (confining_environment[1]*confining_environment[1]))

                radial_distance = fabs(radial_position-target_radial_position)

                print(radial_position , target_radial_position , radial_distance , best_radial_distance , tentative)
                
                # If the midpoint of the segment is closer to the target radial position than the
                # previous guesses. Store the points as the best guesses!
                if radial_distance < best_radial_distance:                    
                    best_radial_distance = radial_distance
                    best_radial_position = radial_position
                    best_tentative       = tentative+1 # The variable tentative starts from 0

                    best_segment_P0 = []
                    best_segment_P1 = []
                    for component_P0 , component_P1 in zip(segment_P0_tmp,segment_P1_tmp):
                        best_segment_P0.append(component_P0)
                        best_segment_P1.append(component_P1)
                    
                tentative = tentative + 1
                
        if best_segment_P0 == []:
            print("Valid placement not found for chromosome rosette after %d tentatives" % tentative)
            sys.exit()

        print("Successfully positioned chromosome of length %lf at tentative %d of %d tentatives" % (length, best_tentative, tentative))        
        segments_P0.append(best_segment_P0)
        segments_P1.append(best_segment_P1)

    print("Successfully generated rod conformation!")
    return segments_P1 , segments_P0
    
##########

def generate_rods_random_conformation(rosettes_lengths, rosette_radius,
                                      confining_environment,
                                      max_number_of_temptative=1000):
    # Construction of the rods initial conformation 
    segments_P0 = []
    segments_P1 = []
    
    for length in rosettes_lengths:
        tentative = 0
        clashes   = 0
        # Random positioning of the rods
        while tentative < max_number_of_temptative and clashes == 0:                

            tentative += 1
            clashes    = 1
            #print "Length = %f" % length

            print("Trying to position terminus 0")
            #pick uniformly within the confining environment using the rejection method 
            first_point = []
            first_point = draw_point_inside_the_confining_environment(confining_environment,
                                                                      rosette_radius,
                                                                      length)

            print("Successfully positioned terminus 0: %f %f %f" % (first_point[0], first_point[1], first_point[2]))
            sys.stdout.flush()
            
            print("Trying to position terminus 1")
            #pick from P0 another point one the sphere of radius length inside the confining environment
            last_point = []
            last_point = draw_second_extreme_of_a_segment_inside_the_confining_environment(first_point[0],
                                                                                           first_point[1],
                                                                                           first_point[2],
                                                                                           length,
                                                                                           rosette_radius,
                                                                                           confining_environment)
            
            print("Successfully positioned terminus 1: %f %f %f" % (last_point[0], last_point[1], last_point[2]))
                
            # Check clashes with the previously positioned rods
            clashes = 1 
            for segment_P1,segment_P0 in zip(segments_P1,segments_P0):
                clashes = check_segments_clashes(segment_P1,
                                                 segment_P0,
                                                 last_point,
                                                 first_point,
                                                 rosette_radius)
                if clashes == 0:
                    break                

            #print clashes
        print("Successfully positioned chromosome of length %lf at tentative %d\n" % (length, tentative))        
        segments_P1.append(last_point)
        segments_P0.append(first_point)            

    print("Successfully generated rod conformation!")
    return segments_P1 , segments_P0

##########

def generate_rods_random_conformation_with_pbc(rosettes_lengths, rosette_radius,
                                               confining_environment,
                                               max_number_of_temptative=100000):

    # Construction of the rods initial conformation 
    segments_P0 = []
    segments_P1 = []
    
    for length in rosettes_lengths:
        tentative = 0
        clashes   = 0
        # Random positioning of the rods
        while tentative < 100000 and clashes == 0:                

            tentative += 1
            clashes    = 1
            #print "Length = %f" % length

            print("Trying to position terminus 0")
            #pick uniformly within the confining environment using the rejection method 
            first_point = []
            first_point = draw_point_inside_the_confining_environment(confining_environment,
                                                                      rosette_radius,
                                                                      length)

            print("Successfully positioned terminus 0: %f %f %f" % (first_point[0], first_point[1], first_point[2]))
            
            print("Trying to position terminus 1")
            #pick from P0 another point one the sphere of radius length inside the confining environment
            last_point = []
            last_point = draw_second_extreme_of_a_segment(first_point[0],
                                                          first_point[1],
                                                          first_point[2],
                                                          length,
                                                          rosette_radius)            
            
            print(last_point)
            # Check clashes with the previously positioned rods
            for segment_P1,segment_P0 in zip(segments_P1,segments_P0):
                clashes = check_segments_clashes_with_pbc(segment_P1,
                                                          segment_P0,
                                                          last_point,
                                                          first_point,
                                                          rosette_radius,
                                                          confining_environment)
                if clashes == 0:
                    break                

            #print clashes
        print("Successfully positioned chromosome of length %lf at tentative %d\n" % (length, tentative))        
        segments_P1.append(last_point)
        segments_P0.append(first_point)            

    print("Successfully generated rod conformation!")
    return segments_P1 , segments_P0

##########

def generate_random_walks(chromosome_particle_numbers,
                          particle_radius,
                          confining_environment,
                          center,
                          pbc=False):
    # Construction of the random walks initial conformation 
    random_walks = []
    
    for number_of_particles in chromosome_particle_numbers:
        #print "Trying to position random walk"

        #print "Positioning first particle"            
        particles_overlap = 0
        while particles_overlap == 0:
            random_walk      = {}
            random_walk['x'] = []
            random_walk['y'] = []
            random_walk['z'] = []        

            particles_overlap = 1

            # Generate the first particle
            first_particle = []
            first_particle = draw_point_inside_the_confining_environment(confining_environment,
                                                                         particle_radius,
                                                                         length)
            random_walk['x'].append(first_particle[0])        
            random_walk['y'].append(first_particle[1])
            random_walk['z'].append(first_particle[2])

            # Generate all the others N-1 particles
            for particle in range(1,number_of_particles):
                #print("Positioning particle %d" % (particle+1))
                sys.stdout.flush()
                particles_overlap = 0 # 0 means that there is an overlap -> PROBLEM
                overlapCounter = -1
                maxIter = 100000000
                overlapCounter += 1
                if overlapCounter > maxIter:
                    # raise error so log file is created to avoid k_seed
                    overlapCounter = -1
                    #errorName = 'ERROR: Initial conformation non ending loop %s' % confining_environment
                    #raise InitalConformationError(errorName)
                particles_overlap = 1
                new_particle = []
                if pbc:
                    new_particle = draw_second_extreme_of_a_segment(
                        random_walk['x'][-1],
                        random_walk['y'][-1],
                        random_walk['z'][-1],
                        2.0*particle_radius,
                        2.0*particle_radius)
                else:
                    new_particle = draw_second_extreme_of_a_segment_inside_the_confining_environment(
                        random_walk['x'][-1],
                        random_walk['y'][-1],
                        random_walk['z'][-1],
                        2.0*particle_radius,
                        2.0*particle_radius,
                        confining_environment)
                random_walk['x'].append(new_particle[0])        
                random_walk['y'].append(new_particle[1])
                random_walk['z'].append(new_particle[2])

            print("Check if there is a particle overlapping with any other particle in the system")
            print("Within random-walks")
            molecule0 = list(zip(random_walk['x'],random_walk['y'],random_walk['z']))
            print(len(molecule0),len(molecule0[0]))
            sys.stdout.flush()
            
            distances = spatial.distance.cdist(molecule0,molecule0)
            print(distances.min())
            """
            for i in range(len(molecule0)):
                for j in range(i+1,len(molecule0)):
                    if distances[(i,j)] < particle_radius*2.0*0.9:
                        print(i,j,distances[(i,j)])
                        particles_overlap = 0
                        if particles_overlap == 0:
                            break
                    if particles_overlap == 0:
                        break
            """
            print("Between Random-walks")
            if particles_overlap != 0:
                for random_walk_pair in list(combinations(random_walks,2)):
                    molecule0 = list(zip(random_walk_pair[0]['x'],random_walk_pair[0]['y'],random_walk_pair[0]['z']))
                    molecule1 = list(zip(random_walk_pair[1]['x'],random_walk_pair[1]['y'],random_walk_pair[1]['z']))
                    distances = spatial.distance.cdist(molecule1,molecule0)
                    print(len(molecule0),len(molecule0[0]),distances.min())
                    sys.stdout.flush()
                    if distances.min() < particle_radius*2.0*0.9:
                        particles_overlap = 0
                        break
            print(particles_overlap)
                
                    
        print("Successfully positioned random walk of %d particles" % number_of_particles)
        random_walks.append(random_walk)

    print("Successfully generated random walk conformation!")
    if center:
        print("Centering random-walk into the origin")
        for random_walk in random_walks:
            x_com, y_com, z_com = (0.0,0.0,0.0)
            cnt = 0
            for (x,y,z) in zip(random_walk['x'],random_walk['y'],random_walk['z']):
                x_com += x
                y_com += y
                z_com += z
                cnt += 1
            x_com, y_com, z_com = (x_com/cnt,y_com/cnt,z_com/cnt)

            for i in range(len(random_walk['x'])):
                random_walk['x'][i] -= x_com
                random_walk['y'][i] -= y_com
                random_walk['z'][i] -= z_com
            
            x_com, y_com, z_com = (0.0,0.0,0.0)
            cnt = 0
            for (x,y,z) in zip(random_walk['x'],random_walk['y'],random_walk['z']):
                x_com += x
                y_com += y
                z_com += z
                cnt += 1
            x_com, y_com, z_com = (x_com/cnt,y_com/cnt,z_com/cnt)
            
    return random_walks

##########

def check_particle_vs_all_overlap(x,y,z,chromosome,overlap_radius):    
    particle_overlap = 1

    for x0, y0, z0 in zip(chromosome['x'],chromosome['y'],chromosome['z']):
        particle_overlap = check_particles_overlap(x0,y0,z0,x,y,z,overlap_radius)
        if particle_overlap == 0:
            return particle_overlap
        
    return particle_overlap
            
##########

def draw_second_extreme_of_a_segment_inside_the_confining_environment(x0, y0, z0, 
                                                                      segment_length, 
                                                                      object_radius, 
                                                                      confining_environment,
                                                                      max_number_of_temptative=10000):
    inside = 0

    temptative = 0

    if confining_environment[0] == 'sphere':
        while inside == 0 and temptative < max_number_of_temptative:
            temptative += 1
            
            particle = []
            temp_theta  = arccos(2.0*random()-1.0)
            temp_phi    = 2*pi*random()
            particle.append(x0 + segment_length * cos(temp_phi) * sin(temp_theta))
            particle.append(y0 + segment_length * sin(temp_phi) * sin(temp_theta))
            particle.append(z0 + segment_length * cos(temp_theta))
            # Check if the particle is inside the confining_environment
            inside = check_point_inside_the_confining_environment(particle[0],
                                                                  particle[1],
                                                                  particle[2],
                                                                  object_radius,
                                                                  confining_environment)
    else:
        while inside == 0 and temptative < max_number_of_temptative:
            temptative += 1
            
            particle = []
            temp_theta  = arccos(2.0*random()-1.0)
            temp_phi    = 2*pi*random()
            particle.append(x0 + segment_length * cos(temp_phi) * sin(temp_theta))
            particle.append(y0 + segment_length * sin(temp_phi) * sin(temp_theta))
            particle.append(z0 + segment_length * cos(temp_theta))
            # Check if the particle is inside the confining_environment
            inside = check_point_inside_the_confining_environment(particle[0],
                                                                  particle[1],
                                                                  particle[2],
                                                                  object_radius,
                                                                  confining_environment)

    return particle

##########

def draw_second_extreme_of_a_segment(x0, y0, z0, 
                                     segment_length, 
                                     object_radius):
    particle = []
    temp_theta  = arccos(2.0*random()-1.0)
    temp_phi    = 2*pi*random()
    particle.append(x0 + segment_length * cos(temp_phi) * sin(temp_theta))
    particle.append(y0 + segment_length * sin(temp_phi) * sin(temp_theta))
    particle.append(z0 + segment_length * cos(temp_theta))
    
    return particle

##########

def draw_point_inside_the_confining_environment(confining_environment, object_radius, length):
    #pick a point uniformly within the confining environment using the rejection method 

    print(confining_environment)
    if confining_environment[0] == 'cube':
        dimension_x = confining_environment[1] * 0.5
        dimension_y = confining_environment[1] * 0.5
        dimension_z = confining_environment[1] * 0.5        
        if len(confining_environment) > 2:
            print("# WARNING: Defined a cubical confining environment with reduntant paramenters.")
            print("# Only 2 are needed the identifier and the side")
        inside = 0
        while inside == 0:
            particle = []
            particle.append((2.0*random()-1.0)*(dimension_x - object_radius))
            particle.append((2.0*random()-1.0)*(dimension_y - object_radius))
            particle.append((2.0*random()-1.0)*(dimension_z - object_radius))
            # Check if the particle is inside the confining_environment
            inside = check_point_inside_the_confining_environment(particle[0],
                                                                  particle[1],
                                                                  particle[2],
                                                                  object_radius,
                                                                  confining_environment)                                
        
    if confining_environment[0] == 'sphere':
        dimension_x = confining_environment[1]-object_radius
        dimension_y = confining_environment[1]-object_radius
        dimension_z = confining_environment[1]-object_radius
        if len(confining_environment) > 2:
            print("# WARNING: Defined a spherical confining environment with reduntant parameters.")
            print("# Only 2 are needed the identifier and the radius")
        
        inside = 0
        while inside == 0:
            particle = []
            
            particle.append((2.0*random()-1.0)*(dimension_x))
            particle.append((2.0*random()-1.0)*(dimension_y))
            particle.append((2.0*random()-1.0)*(dimension_z))
            # Check if the particle is inside the confining_environment
            inside = check_point_inside_the_confining_environment(particle[0],
                                                                  particle[1],
                                                                  particle[2],
                                                                  object_radius,
                                                                  confining_environment)
            print(particle)
            print(object_radius)
                
    if confining_environment[0] == 'ellipsoid':
        if len(confining_environment) < 4:
            print("# ERROR: Defined an ellipsoidal confining environment without the necessary paramenters.")
            print("# 4 are needed the identifier, the x-semiaxes, the y-semiaxes, and the z-semiaxes")
            sys.exit()
        dimension_x = confining_environment[1]
        dimension_y = confining_environment[2]
        dimension_z = confining_environment[3]

    if confining_environment[0] == 'cylinder':
        if len(confining_environment) < 3:
            print("# WARNING: Defined a cylindrical confining environment without the necessary paramenters.")
            print("# 3 are needed the identifier, the basal radius, and the height")
            sys.exit()
        dimension_x = confining_environment[1]
        dimension_y = confining_environment[1]
        dimension_z = confining_environment[2]
            

        
    return particle
    
##########        

def check_point_inside_the_confining_environment(Px, Py, Pz,
                                                 object_radius,
                                                 confining_environment):
    # The shapes are all centered in the origin
    # - sphere    : radius r
    # - cube      : side
    # - cylinder  : basal radius , height
    # - ellipsoid : semi-axes a , b , c ;

    if confining_environment[0] == 'sphere':
        radius = confining_environment[1] - object_radius
        #print(Px,Py,Pz,radius)
        #print(Px,Py,Pz)
        #print((Px*Px)/(radius*radius) + (Py*Py)/(radius*radius) + (Pz*Pz)/(radius*radius))
        if ((Px*Px)/(radius*radius) + (Py*Py)/(radius*radius) + (Pz*Pz)/(radius*radius)) < 1.0 : return 1

    if confining_environment[0] == 'ellipsoid':
        a = confining_environment[1] - object_radius
        b = confining_environment[2] - object_radius
        c = confining_environment[3] - object_radius
        if ((Px*Px)/(a*a) + (Py*Py)/(b*b) + (Pz*Pz)/(c*c)) < 1.0 : return 1

    if confining_environment[0] == 'cube':
        hside = confining_environment[1] * 0.5 - object_radius
        if (((Px*Px)/(hside*hside)) < 1.0) and (((Py*Py)/(hside*hside)) < 1.0) and (((Pz*Pz)/(hside*hside)) < 1.0) : return 1

    if confining_environment[0] == 'cylinder':
        radius      = confining_environment[1]     - object_radius
        half_height = confining_environment[2]*0.5 - object_radius
        if (((Px*Px)/(radius*radius) + (Py*Py)/(radius*radius)) < 1.0) and (((Pz*Pz)/(half_height*half_height)) < 1.0): return 1
            
    return 0

##########

def check_segments_clashes(s1_P1, s1_P0, s2_P1, s2_P0, rosette_radius):

    # Check steric clashes without periodic boundary conditions
    if distance_between_segments(s1_P1, s1_P0, s2_P1, s2_P0) < 2.0*rosette_radius:
        # print "Clash between segments",s1_P1,s1_P0,"and",s2_P1_tmp,s2_P0_tmp,"at distance", distance
        return 0

    return 1

##########

def check_segments_clashes_with_pbc(s1_P1, s1_P0, s2_P1, s2_P0, 
                                    rosette_radius,
                                    confining_environment):

    # Check steric clashes with periodic boundary conditions
    if distance_between_segments(s1_P1, s1_P0, s2_P1, s2_P0) < 2.0*rosette_radius:
        # print "Clash between segments",s1_P1,s1_P0,"and",s2_P1_tmp,s2_P0_tmp,"at distance", distance
        return 0

    return 1

##########

def distance_between_segments(s1_P1, s1_P0, s2_P1, s2_P0):

    # Inspiration: http://softsurfer.com/Archive/algorithm_0106/algorithm_0106.htm 
    # Copyright 2001, softSurfer (www.softsurfer.com)
    # This code may be freely used and modified for any purpose
    # providing that this copyright notice is included with it.
    # SoftSurfer makes no warranty for this code, and cannot be held
    # liable for any real or imagined damage resulting from its use.
    # Users of this code must verify correctness for their application.

    u  = []
    v  = []
    w  = []
    dP = []

    for c_s1_P1,c_s1_P0,c_s2_P1,c_s2_P0 in zip(s1_P1, s1_P0, s2_P1, s2_P0):        
        u.append(c_s1_P1 - c_s1_P0)
        v.append(c_s2_P1 - c_s2_P0)
        w.append(c_s1_P0 - c_s2_P0)
    
    a  = scalar_product(u, u)
    b  = scalar_product(u, v)
    c  = scalar_product(v, v)
    d  = scalar_product(u, w)
    e  = scalar_product(v, w)

    D  = a*c - b*b
    sD = tD = D
        
    if D < (1.0e-7):
        # Segments almost parallel 
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        # Get the closest points on the infinite lines
        sN = (b*e - c*d)
        tN = (a*e - b*d)
        if (sN < 0.0):            
            # sc < 0 => the s=0 edge is visible
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD: # sc > 1 => the s=1 edge is visible
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0: # tc < 0 => the t=0 edge is visible
        tN = 0.0
        # Recompute sc for this edge
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a        
    
    elif tN > tD: # tc > 1 => the t=1 edge is visible
        tN = tD
        # Recompute sc for this edge
        if (-d + b) < 0.0:
            sN = 0
        elif (-d + b) > a:
            sN = sD;
        else:
            sN = (-d + b)
            sD = a

    # Finally do the division to get sc and tc
    if abs(sN) < (1.0e-7):
        sc = 0.0
    else:
        sc = sN / sD
        
    if abs(tN) < (1.0e-7):
        tc = 0.0
    else:
        tc = tN / tD
     
    # Get the difference of the two closest points
    for i in range(len(w)):    
        dP.append(w[i] + ( sc * u[i] ) - ( tc * v[i] )) # = S1(sc) - S2(tc)
    
    return norm(dP)   # return the closest distance

##########

def rosettes_rototranslation(rosettes, segments_P1, segments_P0):

    for i in range(len(segments_P1)):
        vector = []
        theta  = []

        for component_P1 , component_P0 in zip(segments_P1[i], segments_P0[i]):
            vector.append(component_P1-component_P0)
            
        # Rotation Angles
        theta.append(atan2(vector[1],vector[2]))
      
        x_temp_2 =  vector[0]
        y_temp_2 =  cos(theta[0]) * vector[1] - sin(theta[0]) * vector[2]                
        z_temp_2 =  sin(theta[0]) * vector[1] + cos(theta[0]) * vector[2]        
        theta.append(atan2(x_temp_2,z_temp_2))
        
        x_temp_1 =  cos(theta[1]) * x_temp_2 - sin(theta[1]) * z_temp_2
        y_temp_1 =  y_temp_2
        z_temp_1 =  sin(theta[1]) * x_temp_2 + cos(theta[1]) * z_temp_2
        
        if(z_temp_1 < 0.0):            
            z_temp_1 = -z_temp_1
            theta.append(pi)
        else:
            theta.append(0.0)
        #print x_temp_1 , y_temp_1 , z_temp_1 
        
        # Chromosome roto-translations
        for particle in range(len(rosettes[i]['x'])):

            x_temp_2 =   rosettes[i]['x'][particle]
            y_temp_2 =   cos(theta[2]) * rosettes[i]['y'][particle] + sin(theta[2]) * rosettes[i]['z'][particle]
            z_temp_2 = - sin(theta[2]) * rosettes[i]['y'][particle] + cos(theta[2]) * rosettes[i]['z'][particle]
            
            x_temp_1 =   cos(theta[1]) * x_temp_2 + sin(theta[1]) * z_temp_2
            y_temp_1 =   y_temp_2
            z_temp_1 = - sin(theta[1]) * x_temp_2 + cos(theta[1]) * z_temp_2

            x =   x_temp_1;
            y =   cos(theta[0]) * y_temp_1 + sin(theta[0]) * z_temp_1;
            z = - sin(theta[0]) * y_temp_1 + cos(theta[0]) * z_temp_1;
            
            # Chromosome translations
            rosettes[i]['x'][particle] = segments_P0[i][0] + x;
            rosettes[i]['y'][particle] = segments_P0[i][1] + y;
            rosettes[i]['z'][particle] = segments_P0[i][2] + z;
    return rosettes

##########

def scalar_product(a, b):

    scalar = 0.0
    for c_a,c_b in zip(a,b):
        scalar = scalar + c_a*c_b 

    return scalar

##########

def norm(a):

    return sqrt(scalar_product(a, a))

##########

def write_initial_conformation_file(chromosomes,
                                    chromosome_particle_numbers,
                                    confining_environment,
                                    out_file="Initial_conformation.dat",
                                    atom_types=1,
                                    angle_types=1,
                                    bond_types=1):
    # Choosing the appropriate xlo, xhi...etc...depending on the confining environment
    xlim = []
    ylim = []
    zlim = []
    if confining_environment[0] == 'sphere':
        radius = confining_environment[1] + 1.0
        xlim.append(-radius)
        xlim.append(radius)
        ylim.append(-radius)
        ylim.append(radius)
        zlim.append(-radius)
        zlim.append(radius)
        
    if confining_environment[0] == 'ellipsoid':
        a = confining_environment[1] + 1.0
        b = confining_environment[2] + 1.0
        c = confining_environment[3] + 1.0
        xlim.append(-a)
        xlim.append(a)
        ylim.append(-b)
        ylim.append(b)
        zlim.append(-c)
        zlim.append(c)

    if confining_environment[0] == 'cube':
        hside = confining_environment[1] * 0.5
        xlim.append(-hside)
        xlim.append(hside)
        ylim.append(-hside)
        ylim.append(hside)
        zlim.append(-hside)
        zlim.append(hside)
        
    if confining_environment[0] == 'cylinder':
        radius      = confining_environment[1]   + 1.0
        hheight = confining_environment[2] * 0.5 + 1.0
        xlim.append(-radius)
        xlim.append(radius)
        ylim.append(-radius)
        ylim.append(radius)
        zlim.append(-hheight)
        zlim.append(hheight)
    
    fileout = open(out_file,'w')
    n_chr=len(chromosomes)
    n_atoms=0
    for n in chromosome_particle_numbers:
        n_atoms+=n    
        
    fileout.write("LAMMPS input data file \n\n")
    fileout.write("%9d atoms\n" % (n_atoms))
    fileout.write("%9d bonds\n" % (n_atoms-n_chr))
    fileout.write("%9d angles\n\n" % (n_atoms-2*n_chr))
    fileout.write("%9s atom types\n" % atom_types)
    fileout.write("%9s bond types\n" % bond_types)
    fileout.write("%9s angle types\n\n" % angle_types)
    fileout.write("%6.3lf    %6.3lf     xlo xhi\n" % (xlim[0], xlim[1]))
    fileout.write("%6.3lf    %6.3lf     ylo yhi\n" % (ylim[0], ylim[1]))
    fileout.write("%6.3lf    %6.3lf     zlo zhi\n" % (zlim[0], zlim[1]))
  
    fileout.write("\n Atoms \n\n")
    particle_number = 1
    for chromosome in chromosomes:
        for x,y,z in zip(chromosome['x'],chromosome['y'],chromosome['z']):          
            fileout.write("%-8d %s %s %7.4lf %7.4lf %7.4lf\n" % (particle_number, "1", "1", x, y, z))
            particle_number += 1
            
    # for(i = 0; i < N_NUCL; i++)
    # {
    #    k++;
    #    fileout.write("%5d %s %s %7.4lf %7.4lf %7.4lf \n", k, "1", "1", P[i][0], P[i][1], P[i][2]);
    # }
  
    fileout.write("\n Bonds \n\n")
    bond_number          = 1
    first_particle_index = 1
    for chromosome in chromosomes:
        for i in range(len(chromosome['x'])-1):
            fileout.write("%-4d %s %4d %4d\n" % (bond_number, "1", first_particle_index, first_particle_index+1))
            bond_number          += 1
            first_particle_index += 1
        first_particle_index += 1 # I have to go to the end of the chromosome!
          
    fileout.write("\n Angles \n\n")
    angle_number         = 1
    first_particle_index = 1
    for chromosome in chromosomes:        
        for i in range(len(chromosome['x'])-2):
            fileout.write("%-4d %s %5d %5d %5d\n" % (angle_number, "1", first_particle_index, first_particle_index+1, first_particle_index+2))
            angle_number         += 1    
            first_particle_index += 1
        first_particle_index += 2 # I have to go to the end of the chromosome!

    fileout.close()

##########

def distance(x0,y0,z0,x1,y1,z1):
    return sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)+(z0-z1)*(z0-z1))

##########

def check_particles_overlap(x0,y0,z0,x1,y1,z1,overlap_radius):
    if distance(x0,y0,z0,x1,y1,z1) < overlap_radius:
        #print "Particle %f %f %f and particle %f %f %f are overlapping\n" % (x0,y0,z0,x1,y1,z1)
        return 0
    return 1

##########

def store_conformation_with_pbc(xc, result, confining_environment):
    # Reconstruct the different molecules and store them separatelly
    ix    , iy    , iz     = (0, 0, 0)
    ix_tmp, iy_tmp, iz_tmp = (0, 0, 0)
    x_tmp , y_tmp , z_tmp  = (0, 0, 0)

    molecule_number = 0 # We start to count from molecule number 0

    particles = []
    particles.append({})
    particles[molecule_number]['x'] = []
    particles[molecule_number]['y'] = []
    particles[molecule_number]['z'] = []

    particle_counts        = []
    particle_counts.append({}) # Initializing the particle counts for the first molecule
    
    max_bond_length        = (1.5*1.5) # This is the default polymer-based bond length

    for i in range(0,len(xc),3):
        particle = int(i/3)

        x = xc[i]   + ix * confining_environment[1] 
        y = xc[i+1] + iy * confining_environment[1] 
        z = xc[i+2] + iz * confining_environment[1] 
        
        # A - Check whether the molecule is broken because of pbc
        # or if we are changing molecule
        if particle > 0:             
        
            # Compute the bond_length
            bond_length  = (particles[molecule_number]['x'][-1] - x)* \
                           (particles[molecule_number]['x'][-1] - x)+ \
                           (particles[molecule_number]['y'][-1] - y)* \
                           (particles[molecule_number]['y'][-1] - y)+ \
                           (particles[molecule_number]['z'][-1] - z)* \
                           (particles[molecule_number]['z'][-1] - z)
            
            # Check whether the bond is too long. This could mean:
            # 1 - Same molecule disjoint by pbc
            # 2 - Different molecules
            if bond_length > max_bond_length:
                min_bond_length = bond_length                
                x_tmp , y_tmp , z_tmp = (x, y, z)

                # Check if we are in case 1: the same molecule continues 
                # in a nearby box
                indeces_sets = product([-1, 0, 1],
                                       [-1, 0, 1],
                                       [-1, 0, 1])
                
                for (l, m, n) in indeces_sets:
                    # Avoid to check again the same periodic copy
                    if (l, m, n) == (0, 0, 0):
                        continue

                    # Propose a new particle position
                    x = xc[i]   + (ix + l) * confining_environment[1] 
                    y = xc[i+1] + (iy + m) * confining_environment[1] 
                    z = xc[i+2] + (iz + n) * confining_environment[1] 
                    
                    # Check the new bond length
                    bond_length  = (particles[molecule_number]['x'][-1] - x)* \
                                   (particles[molecule_number]['x'][-1] - x)+ \
                                   (particles[molecule_number]['y'][-1] - y)* \
                                   (particles[molecule_number]['y'][-1] - y)+ \
                                   (particles[molecule_number]['z'][-1] - z)* \
                                   (particles[molecule_number]['z'][-1] - z)
                    
                    # Store the periodic copy with the minimum bond length
                    if bond_length < min_bond_length:
                        #print bond_length
                        x_tmp , y_tmp , z_tmp  = (x   , y   , z   )
                        ix_tmp, iy_tmp, iz_tmp = (ix+l, iy+m, iz+n)
                        min_bond_length = bond_length
                
                # If the minimum bond length is yet too large
                # we are dealing with case 2
                if min_bond_length > 10.:
                    # Start another molecule
                    molecule_number += 1

                    particles.append({})
                    particles[molecule_number]['x'] = []
                    particles[molecule_number]['y'] = []
                    particles[molecule_number]['z'] = []


                    particle_counts.append({}) # Initializing the particle counts for the new molecule

                # If the minimum bond length is sufficiently short
                # we are dealing with case 2
                ix, iy, iz = (ix_tmp, iy_tmp, iz_tmp)
                x , y , z  = (x_tmp , y_tmp , z_tmp)

        # To fullfill point B (see below), we have to count how many
        # particle we have of each molecule for each triplet
        # (ix, iy, iz)                
        try:
            particle_counts[molecule_number][(ix, iy, iz)] += 1.0
        except:
            particle_counts[molecule_number][(ix, iy, iz)] = 0.0
            particle_counts[molecule_number][(ix, iy, iz)] += 1.0
            
        particles[molecule_number]['x'].append(x)
        particles[molecule_number]['y'].append(y)
        particles[molecule_number]['z'].append(z)

    # B - Store in the final arrays each molecule in the periodic copy
    # with more particle in the primary cell (0, 0, 0)
    for molecule in range(molecule_number+1):
        max_number = 0
        # Get the periodic box with more particles
        for (l, m, n) in particle_counts[molecule]:
            if particle_counts[molecule][(l, m, n)] > max_number:
                ix, iy, iz = (l, m, n)
                max_number = particle_counts[molecule][(l, m, n)]

        # Translate the molecule to include the largest portion of particles
        # in the (0, 0, 0) image
        for (x, y, z) in zip(particles[molecule]['x'],particles[molecule]['y'],particles[molecule]['z']):
            x = x - ix * confining_environment[1]
            y = y - iy * confining_environment[1]
            z = z - iz * confining_environment[1]

            result['x'].append(x)
            result['y'].append(y)
            result['z'].append(z)


##### Loop extrusion dynamics functions #####
def read_target_loops_input(input_filename, chromosome_length, percentage):
    # Open input file
    fp_input = open(input_filename, "r")

    loops = []
    target_loops = []
    # Get each loop per line and fill the output list of loops
    for line in fp_input.readlines():

        if line.startswith('#'):
            continue

        splitted = line.strip().split()
        loop = []
        loop.append(int(splitted[1]))
        loop.append(int(splitted[2]))

        loops.append(loop)

    #ntarget_loops = int(len(loops)*percentage/100.)    
    ntarget_loops = int(len(loops))    
    shuffle(loops)
    target_loops = loops[0:ntarget_loops]

    return target_loops

########

def read_target_transcription_based_loops_input(input_filename, chromosome_length, percentage):
    # Open input file
    fp_input = open(input_filename, "r")

    loops = []
    target_loops = []
    # Get each loop per line and fill the output list of loops
    for line in fp_input.readlines():

        if line.startswith('#'):
            continue

        splitted = line.strip().split()
        if len(splitted) == 4:
            loop = []
            loop.append(int(splitted[1]))
            loop.append(int(splitted[2]))
            loop.append(int(splitted[3]))
            #print loop
            loops.append(loop)
            #print loops

    #ntarget_loops = int(len(loops)*percentage/100.)    
    ntarget_loops = int(len(loops))    
    #random.shuffle(loops)
    target_loops = loops[0:ntarget_loops]

    print(target_loops)
    return target_loops

##########

def central_loop_extrusion_starting_points(target_loops, chromosome_length):
    initial_loops = []
    # Scroll the target loops and draw a point between each start and stop
    for target_loop in target_loops:

        central_particle =  int((target_loop[0]+target_loop[1]-1)*0.5)

        loop = []
        loop.append(central_particle)
        loop.append(central_particle+1)

        initial_loops.append(loop)

    return initial_loops

##########

def get_maximum_number_of_extrusions(target_loops):
    # The maximum is the maximum nextrusions
    maximum = 0

    for target_loop in target_loops:
        #print initial_loop,target_loop
        
        l = abs(target_loop[2])
        if l > maximum:
            maximum = l

    return maximum

##########

def get_maximum_number_of_extruded_particles(target_loops, initial_loops):
    # The maximum is the maximum sequence distance between a target start/stop particle of a loop
    # and the initial random start/stop of a loop
    maximum = 0
    pair = [0, 0]

    for target_loop,initial_loop in zip(target_loops,initial_loops):
        #print initial_loop,target_loop
        
        l = abs(target_loop[0]-initial_loop[0])+1
        if l > maximum:
            pair = [initial_loop[0], target_loop[0]]
            maximum = l

        l = abs(target_loop[1]-initial_loop[1])+1
        if l > maximum:
            pair = [initial_loop[1], target_loop[1]]
            maximum = l
    print("The maximum segment to extrude is from bead %d to %d and it is %d bead long" % (pair[0], pair[1], maximum))
    return maximum

##########

#def draw_loop_extruder_loading_sites(target_loops, chromosome_length):
#    initial_loops = []
    # Scroll the target loops and draw a point between each start and stop
#    for target_loop in target_loops:

#        random_particle =  randint(target_loop[0], target_loop[1])

#        loop = []
#        loop.append(random_particle)
#        loop.append(random_particle+1)

#        initial_loops.append(loop)

#    return initial_loops

#def draw_loop_extruder_loading_site(chromosome_length,distances):

#    tmp_pair = [0,0]
    
    #print(type(distances))
    # draw a starting point for extrusion along the chromosome
#    pair = choice(list(distances))   
#    while distances[pair] > 2.0:
#        pair = choice(list(distances))

#    tmp_pair[0] = pair[0]
#    tmp_pair[1] = pair[1]       
#    if pair[0] > pair[1]:
#        tmp_pair[0] = pair[1]
#        tmp_pair[1] = pair[0]               
#    print("Choosen",tmp_pair,distances[pair])
#    return tmp_pair

def draw_loop_extruder_loading_site(chromosome_length,distances):

    # draw a starting point for extrusion along the chromosome
    random_particle =  randint(1, chromosome_length-2)

    return [random_particle,random_particle+1]



##########

def get_maximum_number_of_extruded_particles(target_loops, initial_loops):
    # The maximum is the maximum sequence distance between a target start/stop particle of a loop
    # and the initial random start/stop of a loop
    maximum = 0

    for target_loop,initial_loop in zip(target_loops,initial_loops):
        #print initial_loop,target_loop
        
        l = abs(target_loop[0]-initial_loop[0])+1
        if l > maximum:
            maximum = l

        l = abs(target_loop[1]-initial_loop[1])+1
        if l > maximum:
            maximum = l

    return maximum

##########

def compute_particles_distance(xc):
    
    particles = []
    distances = {}

    # Getting the coordinates of the particles
    for i in range(0,len(xc),3):
        x = xc[i]  
        y = xc[i+1]
        z = xc[i+2]
        particles.append((x, y, z))

    # Checking whether the restraints are satisfied
    for pair in combinations(range(len(particles)), 2):
        dist = distance(particles[pair[0]][0],
                        particles[pair[0]][1],
                        particles[pair[0]][2],
                        particles[pair[1]][0],
                        particles[pair[1]][1],
                        particles[pair[1]][2])
        distances[pair] = dist

    return distances

##########

def compute_the_percentage_of_satysfied_restraints(input_file_name,
                                                   restraints,
                                                   output_file_name,
                                                   time_point,
                                                   timesteps_per_k_change):

    ### Change this function to use a posteriori the out.colvars.traj file similar to the obj funct calculation ###
    infile  = open(input_file_name , "r")
    outfile = open(output_file_name, "w")
    if os.path.getsize(output_file_name) == 0:
        outfile.write("#%s %s %s %s\n" % ("timestep","satisfied", "satisfiedharm", "satisfiedharmLowBound"))

    #restraints[pair] = [time_dependent_restraints[time_point+1][pair][0], # Restraint type -> Is the one at time point time_point+1
    #time_dependent_restraints[time_point][pair][1]*10.,                   # Initial spring constant 
    #time_dependent_restraints[time_point+1][pair][1]*10.,                 # Final spring constant 
    #time_dependent_restraints[time_point][pair][2],                       # Initial equilibrium distance 
    #time_dependent_restraints[time_point+1][pair][2],                     # Final equilibrium distance 
    #int(time_dependent_steering_pairs['timesteps_per_k_change']*0.5)]     # Number of timesteps for the gradual change

    # Write statistics on the restraints
    nharm = 0
    nharmLowBound = 0        
    ntot  = 0
    for pair in restraints:
        for i in range(len(restraints[pair][0])):
            if restraints[pair][0][i] == "Harmonic":
                nharm += 1
                ntot  += 1
            if restraints[pair][0][i] == "HarmonicLowerBound":
                nharmLowBound += 1
                ntot  += 1
    outfile.write("#NumOfRestraints = %s , Harmonic = %s , HarmonicLowerBound = %s\n" % (ntot, nharm, nharmLowBound))

    # Memorizing the restraint
    restraints_parameters = {}
    for pair in restraints:
        for i in range(len(restraints[pair][0])):
            #E_hlb_pot_p_106_189
            if restraints[pair][0][i] == "Harmonic":
                name  = "E_h_pot_%d_%d_%d" % (i, int(pair[0])+1, int(pair[1])+1)
            if restraints[pair][0][i] == "HarmonicLowerBound":
                name  ="E_hlb_pot_%d_%d_%d" % (i, int(pair[0])+1, int(pair[1])+1)
            restraints_parameters[name] = [restraints[pair][0][i],
                                           restraints[pair][1][i],
                                           restraints[pair][2][i],
                                           restraints[pair][3][i],
                                           restraints[pair][4][i],
                                           restraints[pair][5][i]]
    #print restraints_parameters
    
    # Checking whether the restraints are satisfied
    columns_to_consider = {}
    for line in infile.readlines():
        nsatisfied             = 0.
        nsatisfiedharm         = 0.
        nsatisfiedharmLowBound = 0.
        ntot                   = 0.
        ntotharm               = 0.
        ntotharmLowBound       = 0.

        line = line.strip().split()        
        
        # Checking which columns contain the pairwise distance
        if line[0][0] == "#":            
            for column in range(2,len(line)):
                # Get the columns with the distances
                if "_pot_" not in line[column]:
                    columns_to_consider[column-1] = line[column]
                    #print columns_to_consider
        else:
            for column in range(1,len(line)):
                if column in columns_to_consider:
                    if column >= len(line):
                        continue
                    dist = float(line[column])
                    
                    # Get which restraints are between the 2 particles
                    for name in ["E_h_pot_"+columns_to_consider[column], "E_hlb_pot_"+columns_to_consider[column]]:
                        if name not in restraints_parameters:
                            #print "Restraint %s not present" % name
                            continue
                        else:
                            pass
                            #print name, restraints_parameters[name] 
                    
                        restrainttype = restraints_parameters[name][0]
                        restraintd0   = float(restraints_parameters[name][3]) + float(line[0])/float(restraints_parameters[name][5])*(float(restraints_parameters[name][4]) - float(restraints_parameters[name][3]))
                        restraintk    = float(restraints_parameters[name][1]) + float(line[0])/float(restraints_parameters[name][5])*(float(restraints_parameters[name][2]) - float(restraints_parameters[name][1]))
                        sqrt_k = sqrt(restraintk)                    
                        limit1 = restraintd0 - 2./sqrt_k
                        limit2 = restraintd0 + 2./sqrt_k

                        if restrainttype == "Harmonic":
                            if dist >= limit1 and dist <= limit2:
                                nsatisfied     += 1.0
                                nsatisfiedharm += 1.0
                                #print "#ESTABLISHED",time_point,name,restraints_parameters[name],limit1,dist,limit2
                            else:
                                pass
                                #print "#NOESTABLISHED",time_point,name,restraints_parameters[name],limit1,dist,limit2
                            ntotharm += 1.0
                        if restrainttype == "HarmonicLowerBound":
                            if dist >= restraintd0:
                                nsatisfied             += 1.0
                                nsatisfiedharmLowBound += 1.0
                                #print "#ESTABLISHED",time_point,name,restraints_parameters[name],dist,restraintd0
                            else:
                                pass
                                #print "#NOESTABLISHED",time_point,name,restraints_parameters[name],dist,restraintd0
                            ntotharmLowBound += 1.0
                        ntot += 1.0
                        #print int(line[0])+(time_point)*timesteps_per_k_change, nsatisfied, ntot, nsatisfiedharm, ntotharm, nsatisfiedharmLowBound, ntotharmLowBound
            if ntotharm         == 0.:
                ntotharm         = 1.0
            if ntotharmLowBound == 0.:
                ntotharmLowBound = 1.0


            outfile.write("%d %lf %lf %lf\n" % (int(line[0])+(time_point)*timesteps_per_k_change, nsatisfied/ntot*100., nsatisfiedharm/ntotharm*100., nsatisfiedharmLowBound/ntotharmLowBound*100.))
    infile.close()
    outfile.close()

##########

def read_objective_function(fname):
    
    obj_func=[]
    fhandler = open(fname)
    line = next(fhandler)
    try:
        while True:
            if line.startswith('#'):
                line = next(fhandler)
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            line_vals = line.split()
            obj_func.append(float(line_vals[1]))
            line = next(fhandler)
    except StopIteration:
        pass
    fhandler.close()        
            
    return obj_func      
##########

def compute_the_objective_function(input_file_name,
                                   output_file_name,
                                   time_point,
                                   timesteps_per_k_change):
    
    
    infile  = open(input_file_name , "r")
    outfile = open(output_file_name, "w")
    if os.path.getsize(output_file_name) == 0:
        outfile.write("#Timestep obj_funct\n")

    columns_to_consider = []

    # Checking which columns contain the energies to sum
    for line in infile.readlines():
        line = line.strip().split()        
        
        # Checking which columns contain the energies to sum
        if line[0][0] == "#":            
            for column in range(len(line)):
                if "_pot_" in line[column]:
                    columns_to_consider.append(column-1)
        else:
            obj_funct = 0.0
            for column in columns_to_consider:
                if column < len(line):
                    obj_funct += float(line[column])
            outfile.write("%d %s\n" % (int(line[0])+timesteps_per_k_change*(time_point), obj_funct))

    infile.close()
    outfile.close()


### get unique list ###
def get_list(input_list):

    output_list = []
    #print(input_list)
    
    for element in input_list:
        #print(type(element))
        if isinstance(element, (int)):
            output_list.append(element)
        if isinstance(element, (list)):
            for subelement in element:
                output_list.append(subelement)
        if isinstance(element, (tuple)):
            for subelement in range(element[0],element[1]+1,element[2]):
                output_list.append(subelement)
    return output_list

########## Apply harmonic restraints
def add_harmonic_restraints(restraints):
    if isinstance(restraints, str):   
        # particle1 particle2 k0 kf d0 df
        fp = open(restraints, "r")

        restrain_number = 0
        for line in fp.readlines():
            
            line = line.strip().split()
            if line[0] == "#":
                continue
            if len(line) != 6:
                print("ERROR in",line,"6 fields expected: particle1 particle2 k_init k_final d_init d_final")


            #bond args = atom1 atom2 Kstart Kstop r0start (r0stop)
            #atom1,atom2 = IDs of 2 atoms in bond
            #Kstart,Kstop = restraint coefficients at start/end of run (energy units)
            #r0start = equilibrium bond distance at start of run (distance units)
            #r0stop = equilibrium bond distance at end of run (optional) (distance units). If not
            #specified it is assumed to be equal to r0start
            lmp.command("fix RESTR%i all restrain bond %i %i %f %f %f %f" % (restrain_number,                                                                                                            
                                                                             int(line[0]),
                                                                             int(line[1]),
                                                                             int(line[2]),
                                                                             int(line[3]),
                                                                             int(line[4]),
                                                                             int(line[5])))
            restrain_number += 1
                    

########## Apply lowerBound harmonic restraints
def add_lowerBound_harmonic_restraints(restraints):

    if isinstance(restraints, str):   
        # particle1 particle2 k0 kf d0 df
        fp = open(restraints, "r")

        restrain_number = 0
        for line in fp.readlines():
            
            line = line.strip().split()
            if line[0] == "#":
                continue
            if len(line) != 6:
                print("ERROR in",line,"6 fields expected: particle1 particle2 k_init k_final d_init d_final")


            #lbound args = atom1 atom2 Kstart Kstop r0start (r0stop)
            #atom1,atom2 = IDs of 2 atoms in bond
            #Kstart,Kstop = restraint coefficients at start/end of run (energy units)
            #r0start = equilibrium bond distance at start of run (distance units)
            #r0stop = equilibrium bond distance at end of run (optional) (distance units). If not
            #specified it is assumed to be equal to r0start
            lmp.command("fix LW_RESTR%i all restrain lbound %i %i %f %f %f %f" % (restrain_number,
                                                                                  int(line[0]),
                                                                                  int(line[1]),
                                                                                  int(line[2]),
                                                                                  int(line[3]),
                                                                                  int(line[4]),
                                                                                  int(line[5])))

            restrain_number += 1
##########
#MPI.Finalize()

