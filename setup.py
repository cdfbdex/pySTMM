from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pystmm',
      version='0.1.0',
      description='Support Tensor Machine Multiclassifier',
      py_modules=["classifier"],
      url='https://github.com/cdfbdex/pySTMM',
      long_description = long_description,
      long_description_content_type="text/markdown",
      author='Carlos Ferrin-Bolaños, Wilfredo Alfonso-Morales and Humberto Loaiza-Correa',
      author_email='cdfbdex@gmail.com',
      license='MIT',
      extras_require = {
            "dev": [
                "pytest >= 3.2",
                "check-manifest",
                "twine",
            ],
      },
      packages=find_packages(),
      install_requires=['mne', 'wget', 'plotly', 'numpy', 'scipy', 'scikit-learn',  'pandas'],
      zip_safe=False
)