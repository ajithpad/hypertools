![Hypertools logo](images/hypercube.png)


"_To deal with hyper-planes in a 14 dimensional space, visualize a 3D space and say 'fourteen' very loudly.  Everyone does it._" - Geoff Hinton


![Hypertools example](images/hypertools.gif)

<h2>Overview</h2>

HyperTools is designed to facilitate
[dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)-based
visual explorations of high-dimensional data.  The basic pipeline is
to feed in a high-dimensional dataset (or a series of high-dimensional
datasets) and, in a single function call, reduce the dimensionality of
the dataset(s) and create a plot.  The package is built atop many
familiar friends, including [matplotlib](https://matplotlib.org/),
[scikit-learn](http://scikit-learn.org/) and
[seaborn](https://seaborn.pydata.org/).  Our package was recently
featured on
[Kaggle's No Free Hunch blog](http://blog.kaggle.com/2017/04/10/exploring-the-structure-of-high-dimensional-data-with-hypertools-in-kaggle-kernels/).

<h2>Try it!</h2>

Click the badge to launch a binder instance with example uses:

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/contextlab/hypertools-paper-notebooks)

or

Check the [repo](https://github.com/ContextLab/hypertools-paper-notebooks) of Jupyter notebooks from the HyperTools [paper](https://arxiv.org/abs/1701.08290).

<h2>Installation</h2>

These instructions assume you have [pip](https://pip.pypa.io/en/stable/installing/) installed on your system.  To install the latest stable version of this package type:

`pip install hypertools`

To install the latest (unstable) version from this repo, first clone the repository using `git`:

`git clone https://github.com/ContextLab/hypertools.git`

Then navigate to the newly created hypertools folder and type:

`pip install .`

Or, to force an upgrade to the latest version type (from within the hypertools folder):

`git pull`

followed by

`pip install --upgrade .`

<h2>Requirements</h2>

This package's dependencies are:
+ python 2.7, 3.4+
+ PPCA>=0.0.2
+ scikit-learn>=0.18.1
+ pandas>=0.18.0
+ seaborn>=0.7.1
+ matplotlib>=1.5.1
+ scipy>=0.17.1
+ numpy>=1.10.4
+ future
+ requests
+ pytest (for development)
+ ffmpeg (for saving animations)

The dependencies should be installed automatically via `pip`.  To install them manually type:

`pip install -r requirements.txt`

<h2>Documentation</h2>

Check out our readthedocs [here](http://hypertools.readthedocs.io/en/latest/).

<h2>Citing</h2>

We wrote a paper about HyperTools, which you can read [here](https://arxiv.org/abs/1701.08290). We also have a repo with example notebooks from the paper [here](https://github.com/ContextLab/hypertools-paper-notebooks).

Please cite as:

`Heusser AC, Ziman K, Owen LLW, Manning JR (2017) HyperTools: A Python toolbox for visualizing and manipulating high-dimensional data.  arXiv: 1701.08290`

Here is a bibtex formatted reference:

```
@ARTICLE {,
    author  = "A C Heusser and K Ziman and L L W Owen and J R Manning",
    title   = "HyperTools: A Python toolbox for visualizing and manipulating high-dimensional data",
    journal = "arXiv",
    year    = "2017",
    volume  = "1701",
    number  = "08290",
    month   = "jan"
}
```

<h2>Contributing</h2>

(Some text borrowed from the [Matplotlib contributing guide](http://matplotlib.org/devdocs/devel/contributing.html).)

<h3>Submitting a bug report</h3>

If you are reporting a bug, please do your best to include the following:

1. A short, top-level summary of the bug. In most cases, this should be 1-2 sentences.
2. A short, self-contained code snippet to reproduce the bug, ideally allowing a simple copy and paste to reproduce. Please do your best to reduce the code snippet to the minimum required.
3. The actual outcome of the code snippet
4. The expected outcome of the code snippet

<h3>Contributing code</h3>

The preferred way to contribute to HyperTools is to fork the main repository on GitHub, then submit a pull request.

+ If your pull request addresses an issue, please use the title to describe the issue and mention the issue number in the pull request description to ensure a link is created to the original issue.

+ All public methods should be documented in the README.

+ Each high-level plotting function should have a simple example in the examples folder. This should be as simple as possible to demonstrate the method.

+ Changes (both new features and bugfixes) should be tested using `pytest`.  Add tests for your new feature to the `tests/` repo folder.

<h2>Testing</h2>

[![Build Status](https://travis-ci.com/ContextLab/hypertools.svg?token=hxjzzuVkr2GZrDkPGN5n&branch=master)](https://travis-ci.com/ContextLab/hypertools)


To test HyperTools, install pytest (`pip install pytest`) and run `pytest` in the HyperTools folder

<h2>Examples</h2>

See [here](http://hypertools.readthedocs.io/en/latest/auto_examples/index.html) for more examples.

<h2>Plot</h2>

```
import hypertools as hyp
hyp.plot(list_of_arrays, 'o', group=list_of_labels)
```

![Plot example](images/plot.gif)

<h2>Align</h2>

```
import hypertools as hyp
aligned_list = hyp.tools.align(list_of_arrays)
hyp.plot(aligned_list)
```

<h3><center>BEFORE</center></h3>

![Align before example](images/align_before.gif)

<h3><center>AFTER</center></h3>

![Align after example](images/align_after.gif)


<h2>Cluster</h2>

```
import hypertools as hyp
hyp.plot(array, 'o', n_clusters=10)
```

![Cluster Example](images/cluster_example.png)


<h2>Describe PCA</h2>

```
import hypertools as hyp
hyp.tools.describe_pca(list_of_arrays)
```
![Describe Example](images/describe_example.png)
