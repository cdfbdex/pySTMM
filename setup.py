from setuptools import setup, find_packages

setup(name='pystmm',
      version='0.1.0',
      description='Support Tensor Machine Multiclassifier',
      url='',
      author='Carlos Ferrin-Bolaños, Wilfredo Alfonso-Morales and Humberto Loaiza-Correa',
      author_email='cdfbdex@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['mne', 'plotly', 'numpy', 'scipy', 'scikit-learn',  'joblib', 'pandas'],
      zip_safe=False)