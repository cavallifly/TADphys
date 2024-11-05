from future import standard_library
standard_library.install_aliases()
from os import environ
from subprocess import Popen, PIPE, check_call, CalledProcessError

from tadphys._version import __version__

# ## Check if we have X display http://stackoverflow.com/questions/8257385/automatic-detection-of-display-availability-with-matplotlib
# if not "DISPLAY" in environ:
#     import matplotlib
#     matplotlib.use('Agg')
# else:
#     try:
#         check_call('python -c "import matplotlib.pyplot as plt; plt.figure()"',
#                    shell=True, stdout=PIPE, stderr=PIPE)
#     except CalledProcessError:
#         import matplotlib
#         matplotlib.use('Agg')

def get_dependencies_version(dico=False):
    """
    Check versions of TADphys and all dependencies, as well and retrieves system
    info. May be used to ensure reproducibility.
    :returns: string with description of versions installed
    """
    versions = {'  TADphys': __version__ + '\n\n'}
    
    try:
        import scipy
        versions['scipy'] = scipy.__version__
    except ImportError:
        versions['scipy'] = 'Not found'
    
    try:
        import numpy
        versions['numpy'] = numpy.__version__
    except ImportError:
        versions['numpy'] = 'Not found'
    try:
        import matplotlib
        versions['matplotlib'] = matplotlib.__version__
    except ImportError:
        versions['matplotlib'] = 'Not found'
    try:
        mcl, _ = Popen(['mcl', '--version'], stdout=PIPE,
                         stderr=PIPE, universal_newlines=True).communicate()
        versions['MCL'] = mcl.split()[1]
    except:
        versions['MCL'] = 'Not found'
   
    try:
        uname, err = Popen(['uname', '-rom'], stdout=PIPE,
                           stderr=PIPE, universal_newlines=True).communicate()
        versions[' Machine'] = uname
    except:
        versions[' Machine'] = 'Not found'

    if dico:
        return versions
    else:
        return '\n'.join(['%15s : %s' % (k, versions[k]) for k in
                          sorted(versions.keys())])


from tadphys.chromosome                 import Chromosome
from tadphys.experiment                 import Experiment, load_experiment_from_reads
from tadphys.chromosome                 import load_chromosome
# from taddyn.modelling.structuralmodels import StructuralModels
# from taddyn.modelling.structuralmodels import load_structuralmodels
# from taddyn.utils.hic_parser         import load_hic_data_from_reads
# from taddyn.utils.hic_parser         import load_hic_data_from_bam
from tadphys.utils.hic_parser         import read_matrix
