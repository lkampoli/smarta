from distutils.core import setup, Extension


def configuration(parent_package='', top_path=None):
      import numpy
      from numpy.distutils.misc_util import Configuration
      from numpy.distutils.misc_util import get_info

      #Necessary for the half-float d-type.
      info = get_info('npymath')

      config = Configuration('',
                             parent_package,
                             top_path)
      config.add_extension('rarfast',
                           ['rarfast.c'],
                           extra_info=info)

      return config

if __name__ == "__main__":
      from numpy.distutils.core import setup
      setup(configuration=configuration)



