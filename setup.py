from setuptools import setup

setup(
    name='spatialtree',
    version='0.1',
    description='Module for building spatial trees (KD, PCA, RP, 2-means, spill)',
    author='Brian McFee',
    author_email='bmcfee@cs.ucsd.edu',
    url='http://www-cse.ucsd.edu/~bmcfee/code/spatialtree/',
    packages=['spatialtree'],
      long_description="""\
        Module for building spatial trees (KD, PCA, RP, 2-means, spill)
      """,
      classifiers=[
          "License :: OSI Approved :: GNU General Public License (GPL)",
          "Programming Language :: Python",
          "Development Status :: 3 - Alpha",
          "Intended Audience :: Developers",
          "Topic :: Utilities",
      ],
      keywords='kd-tree rp-tree nearest-neighbor',
      license='GPL',
      install_requires=[
        'numpy',
        'scipy',
      ],
      )
