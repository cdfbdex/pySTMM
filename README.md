# pySTMM
pySTMM is a Python 3 package implementation for Support Tensor Machine Multiclassifier

The primary target is classification of multivariate biosignals (like EEG, MEG or EMG) Images, and Text.

This code is MIT-licenced.

## Documentation

The documentation is available on URL:

Linear decision function for binary Support Vector Machine may be stated as:

![equation](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20wx%20&plus;%20b)

Multiclassification is reached by OneVsRest, or OneVsOne heuristics.

## Install

#### Using PyPI

```
pip3 install pystmm
```
or using pip+git for the latest version of the code :

```
pip3 install git+https://github.com/cdfbdex/pySTMM/pySTMM
```

Anaconda is not currently supported, if you want to use anaconda, you need to create a virtual environment in anaconda, activate it and use the above command to install it.

#### From sources

For the latest version, you can install the package from the sources using the setup.py script

```
python3 setup.py install
```

or in developer mode to be able to modify the sources.

```
python3 setup.py develop
```

## How to use it

Most of the functions mimic the scikit-learn API, and therefore can be directly used with sklearn. For example, for cross-validation classification of EEG signal using, it is easy as :

```python
import pystmm
from sklearn.model_selection import cross_val_score

# load your data
X = ... # your EEG data, in format Ntrials x Nchannels X Nsamples
y = ... # the labels

# cross validation
clf = pystmm.classifiier.STMM()

accuracy = cross_val_score(clf, cov, y)

print(accuracy.mean())

```

You can also pipeline methods using sklearn Pipeline framework. For example, to classify EEG signal using a SVM classifier in the tangent space, described in [5] :

```python
from pystmm.classifier import STMM
from sklearn.model_selection import cross_val_score

# load your data
X = ... # your EEG data, in format Ntrials x Nchannels X Nsamples
y = ... # the labels

stmm = STMM()

clf = make_pipeline(stmm)
# cross validation
accuracy = cross_val_score(clf, X, y)

print(accuracy.mean())

```

**Check out the example folder for more examples !**


# Contribution Guidelines

The package aims at adopting the [Scikit-Learn](http://scikit-learn.org/stable/developers/contributing.html#contributing-code) and [MNE-Python](http://martinos.org/mne/stable/contributing.html#general-code-guidelines) conventions as much as possible. See their contribution guidelines before contributing to the repository.


# References

> [1] Carlos Ferrin-Bolaños, et.al., "Assessing the Contribution of Covariance Information to the Electroencephalographic Signals of Brain–Computer Interfaces for Spinal Cord Injury Patients ", Revista TecnoLógicas. 2019. DOI: [link](https://doi.org/10.22430/22565337.1392)

# changelog

### v0.1.0.dev
- Add example on EEG Motor-Imagery classification
- Add example on Image classification
- Add example on Text Categorization
- Fix compatibility with scikit-learn v0.24
