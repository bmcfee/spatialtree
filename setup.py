from setuptools import setup

setup(
    name='spatialtree',
    version='0.1',
    description='Module for building spatial spill trees (KD, PCA, RP, 2-means)',
    author='Brian McFee',
    author_email='bmcfee@cs.ucsd.edu',
    url='http://www-cse.ucsd.edu/~bmcfee/code/spatialtree/',
    packages=['spatialtree'],
      long_description="""\
        Module for building spatial spill trees.

        Supported tree types:
            * KD (maximum-variance)
            * PCA (PD, principal direction)
            * Random projection
            * 2-means
      """,
      classifiers=[
          "License :: OSI Approved :: GNU General Public License (GPL)",
          "Programming Language :: Python",
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Topic :: Data structures",
      ],
      keywords='kd-tree rp-tree nearest-neighbor',
      license='GPL',
      install_requires=[
        'numpy',
        'scipy',
      ],
      )
