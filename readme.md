# Python resources to do stuffs related to atmospheric science

This curated list contains awesome open-source projects with a focus primarily on disciplines related to Atmospheric science. If you like to add or update projects, feel free to open an issue, submit a pull request, or directly edit the readme.md. Contributions are very welcome!

[**Click this**](tutorial.md) for links of useful python tutorial videos.

# Basic python
- [numpy](https://numpy.org/): A fundamental package for MATLAB like array computing in Python
- [pandas](https://pandas.pydata.org/): An open-source library that is made mainly for working with relational or labeled data both easily and intuitively.
- [scipy](https://scipy.org/): SciPy provides algorithms for optimization, integration, interpolation, eigenvalue problems, algebraic equations, differential equations, statistics and many other classes of problems.
- [xarray](https://docs.xarray.dev/en/stable/index.html): It is an indispensible library for working with NetCDF, GRIB, raster, hdf and similar datasets. Xarray makes working with labelled multi-dimensional arrays in Python simple, efficient, and fun!
- [sympy](https://github.com/sympy/sympy): A python library for symbolic mathematics

# Visualization
- [matplotlib](https://matplotlib.org/stable/index.html): Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
- [proplot](https://github.com/proplot-dev/proplot): A succinct matplotlib wrapper for making beautiful, publication-quality graphics
- [Faceted](https://github.com/spencerkclark/faceted): A python library for plotting publication quality plots 
- [Basemap](https://matplotlib.org/basemap/api/basemap_api.html): Plot geospatial data on map projections (with coastlines and political boundaries) using matplotlib. Note that the support for basemap in python have ended. So, it's better to switch to cartopy for visualizations on map.
- [cartopy](https://pypi.org/project/Cartopy/): Cartopy is a Python package designed to make drawing maps for data analysis and visualisation easy.
- [Holoviews](https://holoviews.org/getting_started/Gridded_Datasets.html): A library for interactive plots like D3.js in JavaScript 
- [Geoviews](https://geoviews.org/): A library for creating interactive maps. This libray makes it easy to explore and visualize geographical, meteorological, and oceanographic datasets, such as those used in weather, climate, and remote sensing research. 
- [psyplot](https://psyplot.github.io/): Interactive Data Visualization from Python and GUIs (**especially for ICON model data**). It provides ncview like terminal interface for exploring and visualizing the geospatial data.
- [Bokeh](https://github.com/bokeh/bokeh): Interactive Data Visualization in the browser, from Python
- [Graphviz](https://github.com/xflr6/graphviz): Simple Python interface for plotting graphs (nodes and arrows)
- [hvPlot](https://github.com/holoviz/hvplot): A high-level plotting API for pandas, dask, xarray, and networkx
- [PyVista](https://github.com/pyvista/pyvista): 3D plotting and mesh analysis
- [VisPy](https://github.com/vispy/vispy): High-performance interactive 2D/3D data visualization library
- [mpl3](https://github.com/mpld3/mpld3): A D3 Viewer for Matplotlib
- [arviz](https://github.com/arviz-devs/arviz): Exploratory analysis of Bayesian models with Python
# Statistics
- [statsmodels](https://www.statsmodels.org/stable/index.html): A must have library for statistical modeling and inference.
- [Seaborn](https://github.com/mwaskom/seaborn): Statistical data visualization in Python
- [Altair](https://github.com/altair-viz/altair):  Declarative statistical visualization library for Python
- [PyMC3](https://github.com/pymc-devs/pymc): Bayesian Modeling in Python
- [tensorflow-probability](https://github.com/tensorflow/probability): Probabilistic reasoning and statistical analysis in Tensorflow
- [pyro](https://github.com/pyro-ppl/pyro): Deep universal probabilistic programming with Python and PyTorch
- [pomegranate](https://github.com/jmschrei/pomegranate): Fast, flexible and easy to use probabilistic modelling in Python e.g. **computation of partial correlation**
- [hmmlearn](https://github.com/hmmlearn/hmmlearn): Hidden Markov Models in Python, with scikit-learn like API
- [filterpy](https://github.com/rlabbe/filterpy): Python Kalman filtering and optimal estimation library.
- [GPflow](https://github.com/GPflow/GPflow): Gaussian processes in TensorFlow
- [Orbit](https://github.com/uber/orbit): a Python package for **Bayesian time series forecasting and inference**
- [patsy](https://github.com/pydata/patsy): Describing statistical models in Python using symbolic formulas. Patsy brings the convenience of R "formulas" to Python.
- [bambi](https://github.com/bambinos/bambi): BAyesian Model-Building Interface (Bambi) in Python.

# Geospatial data
- [GeoPandas](https://github.com/geopandas/geopandas): Python tool for working with geographical vector data 
- [folium](https://github.com/python-visualization/folium): Plotting on interacive maps like leaflet
- [rasterio](https://github.com/rasterio/rasterio): Rasterio reads and writes geospatial raster datasets
- [shapely](https://github.com/shapely/shapely): Manipulation and analysis of geometric objects
- [pyproj](https://github.com/pyproj4/pyproj): Python interface to PROJ (cartographic projections and coordinate.
- [Fiona](https://github.com/Toblerity/Fiona): Fiona reads and writes geographic data files
- [geojson](https://github.com/jazzband/geojson):  Python bindings and utilities for GeoJSON

# Atmospheric science stuffs
- [metpy](https://github.com/Unidata/MetPy): MetPy is a collection of tools in Python for reading, visualizing and performing calculations with weather data.
- [cfgrib](https://github.com/ecmwf/cfgrib): A Python interface to map GRIB files to the NetCDF Common Data Model following the CF Convention using ecCodes
- 

# Parallel computing
- [dask](https://github.com/dask/dask): Parallel computing with task scheduling
- [mpi4py](https://github.com/mpi4py/mpi4py/): This package provides Python bindings for the Message Passing Interface (MPI) standard.
- [joblib](https://github.com/joblib/joblib): Joblib provides a simple helper class to write parallel for loops using multiprocessing.

# Working with models
- [PyMieScatt](https://github.com/bsumlin/PyMieScatt): A collection of forward and inverse Mie solving routines for Python 3, based on Bohren and Huffman's Mie Theory derivations
- [typhon](https://github.com/atmtools/typhon): Tools for atmospheric research
- [climt](https://github.com/CliMT/climt): a Python based climate modelling toolkit.
- [lowtran](https://github.com/space-physics/lowtran): LOWTRAN atmospheric absorption extinction, scatter and irradiance model--in Python 
- [xESMF](https://github.com/JiaweiZhuang/xESMF): Universal Regridder for Geospatial Data

# Atmospheric Chemistry
- [AtChem2](https://github.com/AtChem/AtChem2): Atmospheric chemistry box-model for the MCM
- [AC_tools](https://github.com/tsherwen/AC_tools): Atmospheric Chemistry Tools (AC_Tools) contains functions and scripts used for working with atmospheric model output and observational data
- [PyCHAM](https://github.com/simonom/PyCHAM): PyCHAM: CHemistry with Aerosol Microphysics in Python box model 
- [PySDM](https://github.com/open-atmos/PySDM): Pythonic particle-based (super-droplet) warm-rain/aqueous-chemistry cloud microphysics package with box, parcel & 1D/2D prescribed-flow examples
- [PyBox](https://github.com/loftytopping/PyBox): PyBox is a Python based box-model generator and simulator designed for atmospheric chemistry and aerosol studies.

# Time Series Data
Visit [this link](https://github.com/MaxBenChrist/awesome_time_series_in_python) for more comprehensive coverage.

- [darts](https://github.com/unit8co/darts): A python library for user-friendly forecasting and anomaly detection on time series.
- [DLMMC](https://github.com/justinalsing/dlmmc): Dynamical linear modeling (DLM) regression code for analysis of atmospheric time-series data
- [traces](https://github.com/datascopeanalytics/traces): A Python library for unevenly-spaced time series analysis

# AI/ML
Visit [this link](https://github.com/ml-tooling/best-of-ml-python) for more comprehensive coverage.

- [scikit-learn](https://scikit-learn.org/stable/index.html): scikit-learn is a Python module for machine learning
- [tensorflow](https://www.tensorflow.org/): Developed by Google, python library for creating Deep Learning models
- [pytorch](https://pytorch.org/): Developed by Facebook, another very popular deep learning library
- [keras](https://keras.io/): Keras is an open-source library that provides a Python interface for artificial neural networks. Keras acts as an interface for the TensorFlow library
- [Keras Tuner](https://github.com/keras-team/keras-tuner): A Hyperparameter Tuning Library for Keras
- [hyperopt](https://github.com/hyperopt/hyperopt): Distributed Asynchronous Hyperparameter Optimization in Python
- [hyperas](https://github.com/maxpumperla/hyperas): Keras + Hyperopt: A very simple wrapper for convenient
- [Bayesian Optimization](https://github.com/bayesian-optimization/BayesianOptimization): A Python implementation of global optimization with gaussian processes
- [BoTorch](https://github.com/pytorch/botorch): Bayesian optimization in PyTorch
- [Dragonfly](https://github.com/dragonfly/dragonfly): An open source python library for scalable Bayesian optimisation
- [Talos](https://github.com/autonomio/talos): Hyperparameter Optimization for TensorFlow, Keras and PyTorch
- [AutoKeras](https://github.com/keras-team/autokeras): AutoML library for deep learning
- [auto-sklearn](https://github.com/automl/auto-sklearn): Automated Machine Learning with scikit-learn
- [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn): A Python Package to Tackle the Curse of Imbalanced datasets
- [scikit-opt](https://github.com/guofei9987/scikit-opt): Genetic Algorithm, Particle Swarm Optimization, Simulated Annealing

# ML Model interpretability
Libraries to visualize, explain, debug, evaluate, and interpret machine learning models. Visit [this link](https://github.com/ml-tooling/best-of-ml-python) for more comprehensive coverage.

- [xai](https://github.com/EthicalML/xai): An eXplainability toolbox for machine learning
- [shap](https://github.com/slundberg/shap): A game theoretic approach to explain the output of any machine learning model
- [dtreeviz](https://github.com/parrt/dtreeviz): A python library for decision tree visualization and model interpretation
- [explainerdashboard](https://github.com/oegedijk/explainerdashboard): This package makes it convenient to quickly deploy a dashboard web app that explains the workings of a (scikit-learn compatible) machine learning model. The dashboard provides interactive plots on model performance, feature importances, feature contributions to individual predictions, "what if" analysis, partial dependence plots, SHAP (interaction) values, visualisation of individual decision trees, etc.
- [Keract](https://github.com/philipperemy/keract#keract-keras-activations--gradients): Layers Outputs and Gradients in Keras. Made easy.
- [DiCE](https://github.com/interpretml/DiCE): Generate Diverse Counterfactual Explanations for any machine learning model.
- [tf-explain](https://github.com/sicara/tf-explain): Interpretability Methods for tf.keras models
- [explainx](https://github.com/explainX/explainx): ExplainX is a model explainability/interpretability framework for data scientists
- [keras-vis](https://github.com/raghakot/keras-vis): Neural network visualization toolkit for keras
- [flashtorch](https://github.com/MisaOgura/flashtorch): Visualization toolkit for neural networks in PyTorch

# Causal Learning
- [tigramite](https://github.com/jakobrunge/tigramite): A python package for causal inference with a focus on time series data
- [DoWhy](https://github.com/py-why/dowhy): Developed by Microsoft, DoWhy is a Python library for causal inference
- [CausalNex](https://github.com/quantumblacklabs/causalnex): CausalNex aims to become one of the leading libraries for causal reasoning and "what-if" analysis using Bayesian Networks.

# AI/ML for atmospheric science
- [Deep Convolutional AutoEncoder](https://github.com/Opio-Cornelius/deep_convolutional_autoencoder): This repository is for convolutional autoencoder algorithms that can be used to bias correct and analyze output from a numerical model. The algorithms used here were tested on the WRF-chem model for bias correcting simulations of Nitorgen dioxide (NO2), Carbon monoxide (CO), Rainfall and Temperature. 
- 

