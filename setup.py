from setuptools import setup
from distutils.core import setup, Extension

def main():
    # c++ module to compute the distance matrix of single model
    squared_distance_matrix_module = Extension('tadphys.squared_distance_matrix',
                                               language = "c++",
                                               runtime_library_dirs=['3d-lib/'],
                                               sources=['src/3d-lib/squared_distance_matrix_calculation_py.c'],
                                               extra_compile_args=["-ffast-math"])

    setup(
        name         = 'TADphys',
        version      = '0.1',
        author       = 'Marco Di Stefano',
        author_email = 'marco.di.distefano.1985@gmail.com',
        ext_modules  = [squared_distance_matrix_module],
        packages     = ['tadphys', 'tadphys.utils',
                        'tadphys.modelling'],
        platforms = "OS Independent",
        license = "GPLv3",
        description  = 'Tadphys is a Python library that allows to model and explore single or time-series 3C-based data.',
        long_description = (open("README.rst").read()),
        #url          = 'https://github.com/3DGenomes/tadbit',
        #download_url = 'https://github.com/3DGenomes/tadbit/tarball/master',
    )

if __name__ == '__main__':

    exit(main())
