# Python resources to do stuffs related to atmospheric science

This curated list contains awesome open-source projects with a focus primarily on disciplines related to Atmospheric science. If you like to add or update projects, feel free to open an issue, submit a pull request, or directly edit the readme.md. Contributions are very welcome!

[**Check this**](tutorial.md) for links of useful python tutorial videos.

# Basic python
- [numpy](https://numpy.org/): A fundamental package for MATLAB like array computing in Python
- [pandas](https://pandas.pydata.org/): An open-source library that is made mainly for working with relational or labeled data both easily and intuitively.
- [scipy](https://scipy.org/): SciPy provides algorithms for optimization, integration, interpolation, eigenvalue problems, algebraic equations, differential equations, statistics and many other classes of problems.
- [xarray](https://docs.xarray.dev/en/stable/index.html): It is an **indispensible library** for working with NetCDF, GRIB, raster, hdf and similar datasets. Xarray makes working with labelled multi-dimensional arrays in Python simple, efficient, and fun!
- [sympy](https://github.com/sympy/sympy): A python library for symbolic mathematics
- [cupy](https://github.com/cupy/cupy): NumPy & SciPy for GPU
- [pint-xarray](https://github.com/xarray-contrib/pint-xarray): Handling units in xarray
- [XrViz](https://github.com/intake/xrviz): an interactive graphical user interface(GUI) for visually browsing Xarrays.
- [salem](https://github.com/fmaussion/salem): Add geolocalised subsetting, masking, and plotting operations to xarray
- [modin](https://github.com/modin-project/modin): Scale your pandas workflows by changing one line of code
- [vaex](https://github.com/vaexio/vaex): A high performance Python library for lazy Out-of-Core DataFrames (similar to Pandas), to visualize and explore big tabular datasets. 

# Visualization
- [matplotlib](https://matplotlib.org/stable/index.html): Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
- [Science Plots](https://github.com/garrettj403/SciencePlots): Matplotlib styles for scientific figures
- [proplot](https://github.com/proplot-dev/proplot): A succinct matplotlib wrapper for making beautiful, publication-quality graphics
- [Faceted](https://github.com/spencerkclark/faceted): A python library for plotting publication quality plots 
- [gif](https://github.com/maxhumber/gif): Making **GIFs** easily in python
- [xmovie](https://github.com/jbusecke/xmovie): A simple way of creating movies from xarray objects
- [Colormaps](https://github.com/bradyrx/climate_in_color): Tutorial on building and using effective colormaps in climate science
- [Basemap](https://matplotlib.org/basemap/api/basemap_api.html): Plot geospatial data on map projections (with coastlines and political boundaries) using matplotlib. Note that the support for basemap in python have ended. So, it's better to switch to cartopy for visualizations on map.
- [cartopy](https://pypi.org/project/Cartopy/): Cartopy is a Python package designed to make drawing maps for data analysis and visualisation easy.
- [Holoviews](https://holoviews.org/getting_started/Gridded_Datasets.html): A library for interactive plots like D3.js in JavaScript 
- [Geoviews](https://geoviews.org/): A library for creating interactive maps. This libray makes it easy to explore and visualize geographical, meteorological, and oceanographic datasets, such as those used in weather, climate, and remote sensing research. 
- [psyplot](https://psyplot.github.io/): Interactive Data Visualization from Python and GUIs (**especially for ICON model data**). It provides **ncview like terminal interface** for exploring and visualizing the geospatial data.
- [geoplot](https://github.com/ResidentMario/geoplot): a high-level Python geospatial plotting library.
- [Bokeh](https://github.com/bokeh/bokeh): Interactive Data Visualization in the browser, from Python
- [Graphviz](https://github.com/xflr6/graphviz): Simple Python interface for plotting graphs (nodes and arrows)
- [hvPlot](https://github.com/holoviz/hvplot): A high-level plotting API for pandas, dask, xarray, and networkx
- [PyVista](https://github.com/pyvista/pyvista): 3D plotting and mesh analysis
- [VisPy](https://github.com/vispy/vispy): High-performance interactive 2D/3D data visualization library
- [mpl3](https://github.com/mpld3/mpld3): A D3 Viewer for Matplotlib
- [arviz](https://github.com/arviz-devs/arviz): Exploratory analysis of Bayesian models with Python

# Statistics
- [statsmodels](https://www.statsmodels.org/stable/index.html): A must have library for statistical modeling and inference.
- [Linear Models](https://github.com/bashtage/linearmodels): Linear (regression) models for Python. Extends statsmodels with Panel regression, instrumental variable estimators etc.
- [Prince](https://github.com/MaxHalford/prince): Multivariate exploratory data analysis (like PCA) in Python
- [Seaborn](https://github.com/mwaskom/seaborn): Statistical data visualization in Python
- [Altair](https://github.com/altair-viz/altair):  Declarative statistical visualization library for Python
- [Skill Metrics](https://github.com/PeterRochford/SkillMetrics): library for calculating and displaying the skill of model predictions against observations such as **Taylor Diagram**
- [PyMC3](https://github.com/pymc-devs/pymc): Bayesian Modeling in Python
- [Pingouin](https://github.com/raphaelvallat/pingouin): Pingouin is designed for users who want simple yet exhaustive statistical functions e.g. **computation of partial correlation**
- [scikits-bootstrap](https://github.com/cgevans/scikits-bootstrap): Python/numpy bootstrap confidence interval estimation.
- [bayesian_bootstrap](https://github.com/lmc2179/bayesian_bootstrap): Bayesian bootstrapping in Python
- [tensorflow-probability](https://github.com/tensorflow/probability): Probabilistic reasoning and statistical analysis in Tensorflow
- [pyro](https://github.com/pyro-ppl/pyro): Deep universal probabilistic programming with Python and PyTorch
- [hmmlearn](https://github.com/hmmlearn/hmmlearn): Hidden Markov Models in Python, with scikit-learn like API
- [filterpy](https://github.com/rlabbe/filterpy): Python Kalman filtering and optimal estimation library.
- [GPflow](https://github.com/GPflow/GPflow): Gaussian processes in TensorFlow
- [Orbit](https://github.com/uber/orbit): a Python package for **Bayesian time series forecasting and inference**
- [patsy](https://github.com/pydata/patsy): Describing statistical models in Python using symbolic formulas. Patsy brings the convenience of R "formulas" to Python.
- [bambi](https://github.com/bambinos/bambi): BAyesian Model-Building Interface (Bambi) in Python.
- [pyextremes](https://github.com/georgebv/pyextremes): Extreme Value Analysis (EVA) in Python
- [confidence interval](https://github.com/jacobgil/confidenceinterval): a package that computes common machine learning metrics like F1, and returns their confidence intervals
- [PyWavelets](https://github.com/PyWavelets/pywt): Wavelet Transforms in Python
- [Impyute](https://github.com/eltonlaw/impyute): a library of missing data imputation algorithms.
- [hoggorm](https://github.com/olivertomic/hoggorm): Explorative multivariate statistics in Python like PCR (principal component regression), PLSR (partial least squares regression)

# Geospatial data
- [GeoPandas](https://github.com/geopandas/geopandas): Python tool for working with geographical vector data 
- [folium](https://github.com/python-visualization/folium): Plotting on interacive maps like leaflet
- [rasterio](https://github.com/rasterio/rasterio): Rasterio reads and writes geospatial raster datasets
- [shapely](https://github.com/shapely/shapely): Manipulation and analysis of geometric objects
- [pyproj](https://github.com/pyproj4/pyproj): Python interface to PROJ (cartographic projections and coordinate.
- [Fiona](https://github.com/Toblerity/Fiona): Fiona reads and writes geographic data files
- [geojson](https://github.com/jazzband/geojson):  Python bindings and utilities for GeoJSON
- [GeoTile](https://github.com/iamtekson/geotile): The python library for tiling the geographic raster data (eg. Tiff etc)
- [nctoolkit](https://github.com/pmlmodelling/nctoolkit): Fast and easy analysis of netCDF data in Python
- [sklearn-xarray](https://github.com/phausamann/sklearn-xarray): The package contains wrappers that allow the user to apply scikit-learn estimators to xarray types without losing their labels.
- [Earthpy](https://github.com/earthlab/earthpy): EarthPy makes it easier to plot and manipulate spatial data in Python.

# Atmospheric science stuffs
- [metpy](https://github.com/Unidata/MetPy): MetPy is a collection of tools in Python for reading, visualizing and performing calculations with weather data.
- [cfgrib](https://github.com/ecmwf/cfgrib): A Python interface to map GRIB files to the NetCDF Common Data Model following the CF Convention using ecCodes
- [xcast](https://github.com/kjhall01/xcast): A Climate Forecasting Toolkit designed to help forecasters and earth scientists apply state-of-the-art postprocessing techniques to gridded data sets.
- [scikit-downscale](https://github.com/pangeo-data/scikit-downscale): Statistical climate downscaling in Python
- [uxarray](https://github.com/UXARRAY/uxarray): Xarray-styled package for reading and directly operating on unstructured grid datasets
- [Metview](https://github.com/ecmwf/metview-python): Python interface to Metview meteorological workstation and batch system
- [xMCA](https://github.com/nicrie/xmca): Maximum Covariance Analysis in Python
- [ConTrack - Contour Tracking](https://github.com/steidani/ConTrack): Contour Tracking of circulation anomalies (atmospheric blocking, cyclones and anticyclones) in weather and climate data
- [WaveBreaking](https://github.com/skaderli/WaveBreaking): Detect, classify, and track Rossby Wave Breaking (RWB) in weather and climate data.
- [climateforcing](https://github.com/chrisroadmap/climateforcing): Tools for analysis of climate model data
- [Atlite](https://github.com/PyPSA/atlite): A Lightweight Python Package for **Calculating Renewable Power Potentials** and Time Series
- [**Access Cmip6**](https://github.com/TaufiqHassan/acccmip6): Python package for accessing and downloading CMIP6 database

# Parallel computing
- [dask](https://github.com/dask/dask): Parallel computing with task scheduling
- [multiprocessing](https://docs.python.org/3/library/multiprocessing.html): Process-based parallelism
- [mpi4py](https://github.com/mpi4py/mpi4py/): This package provides Python bindings for the Message Passing Interface (MPI) standard.
- [joblib](https://github.com/joblib/joblib): Joblib provides a simple helper class to write parallel for loops using multiprocessing.

# Working with models
- [PyMieScatt](https://github.com/bsumlin/PyMieScatt): A collection of forward and inverse Mie solving routines for Python 3, based on Bohren and Huffman's Mie Theory derivations
- [PyTMatrix](https://github.com/jleinonen/pytmatrix): Python code for T-matrix scattering calculations
- [typhon](https://github.com/atmtools/typhon): Tools for atmospheric research
- [climt](https://github.com/CliMT/climt): a Python based climate modelling toolkit.
- [lowtran](https://github.com/space-physics/lowtran): LOWTRAN atmospheric absorption extinction, scatter and irradiance model--in Python 
- [xESMF](https://github.com/JiaweiZhuang/xESMF): Universal Regridder for Geospatial Data
- [gt4py](https://github.com/GridTools/gt4py): Python library for generating high-performance implementations of stencil kernels for weather and climate modeling from a domain-specific language
- [pace](https://github.com/ai2cm/pace): Pace is an implementation of the FV3GFS / SHiELD atmospheric model developed by NOAA/GFDL using the GT4Py domain-specific language in Python. 
- [konrad](https://github.com/atmtools/konrad): konrad is a one-dimensional radiative-convective equilibrium (RCE) model. 
- [pyLRT](https://github.com/EdGrrr/pyLRT): A simple python interface/wrapper for [LibRadTran](http://www.libradtran.org/doku.php)
- [pyClimat](https://github.com/Dan-Boat/pyClimat): a python package for analysising GCM model output and visualization
- [pyLBL](https://github.com/GRIPS-code/pyLBL): Python line-by-line radiative transfer model
- [climlab](https://github.com/climlab/climlab): Python package for process-oriented climate modeling

# Atmospheric Chemistry
- [AtChem2](https://github.com/AtChem/AtChem2): Atmospheric chemistry box-model for the MCM
- [AC_tools](https://github.com/tsherwen/AC_tools): Atmospheric Chemistry Tools (AC_Tools) contains functions and scripts used for working with atmospheric model output and observational data
- [PyCHAM](https://github.com/simonom/PyCHAM): PyCHAM: CHemistry with Aerosol Microphysics in Python box model 
- [PySDM](https://github.com/open-atmos/PySDM): Pythonic particle-based (super-droplet) warm-rain/aqueous-chemistry cloud microphysics package with box, parcel & 1D/2D prescribed-flow examples
- [PyBox](https://github.com/loftytopping/PyBox): PyBox is a Python based box-model generator and simulator designed for atmospheric chemistry and aerosol studies.
- [pykpp](https://github.com/barronh/pykpp): pykpp is a KPP-like chemical mechanism parser that produces a box model solvable by SciPy's odeint solver

# Time Series Data
Visit [this link](https://github.com/MaxBenChrist/awesome_time_series_in_python) for more comprehensive coverage.

- [sktime](https://github.com/alan-turing-institute/sktime): A unified framework for machine learning with time series.
- [Statistical Forecast](https://github.com/Nixtla/statsforecast): Lightning fast forecasting with statistical and econometric models
- [Machine Learning Forecast](https://github.com/Nixtla/mlforecast): Scalable machine learning for time series forecasting
- [Neural Forecast](https://github.com/Nixtla/neuralforecast): User friendly state-of-the-art neural forecasting models.
- [darts](https://github.com/unit8co/darts): A python library for user-friendly forecasting and anomaly detection on time series.
- [DLMMC](https://github.com/justinalsing/dlmmc): Dynamical linear modeling (DLM) regression code for analysis of atmospheric time-series data
- [traces](https://github.com/datascopeanalytics/traces): A Python library for unevenly-spaced time series analysis
- [tsai](https://github.com/timeseriesAI/tsai): State-of-the-art Deep Learning library for Time Series and Sequences.
- [bayesloop](https://github.com/christophmark/bayesloop): Fitting time series models with time-varying parameters and model selection based on Bayesian inference.

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
- [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet): Latex and python code for making neural networks diagrams
- [x-unet](https://github.com/lucidrains/x-unet): Implementation of a U-net complete with efficient attention
- [dvc](https://github.com/iterative/dvc): Data Version Control | Git for Data & Models | ML Experiments Management

# ML Model interpretability
Libraries to visualize, explain, debug, evaluate, and interpret machine learning models. Visit [this link](https://github.com/ml-tooling/best-of-ml-python) for more comprehensive coverage.

- [xai](https://github.com/EthicalML/xai): An eXplainability toolbox for machine learning
- [shap](https://github.com/slundberg/shap): A game theoretic approach to explain the output of any machine learning model
- [PiML](https://github.com/SelfExplainML/PiML-Toolbox): Python toolbox for interpretable machine learning model development and validation.
- [Xplique](https://github.com/deel-ai/xplique): a Python toolkit dedicated to explainability, currently based on Tensorflow
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
- [causal-curve](https://github.com/ronikobrosly/causal-curve): A python package with tools to perform causal inference using observational data when the treatment of interest is continuous.
- [Causal ML](https://github.com/uber/causalml):Causal inference with machine learning algorithms
- [y0](https://github.com/y0-causal-inference/y0): y0 (pronounced "why not?") is Python code for causal inference.
- [skccm](https://github.com/nickc1/skccm): Convergent Cross Mapping in Scikit Learn's style. Convergent Cross Mapping can be used as a way to detect causality between time series.
- [causallib](https://github.com/BiomedSciAI/causallib): A Python package for modular causal inference analysis and model evaluations

# AI/ML for atmospheric science
- [Deep Convolutional AutoEncoder](https://github.com/Opio-Cornelius/deep_convolutional_autoencoder): This repository is for convolutional autoencoder algorithms that can be used to bias correct and analyze output from a numerical model. The algorithms used here were tested on the WRF-chem model for bias correcting simulations of Nitorgen dioxide (NO2), Carbon monoxide (CO), Rainfall and Temperature. 
- [Atmos. Chem. Downscaling CNN](https://github.com/avgeiss/chem_downscaling): Downscaling Atmospheric Chemistry Simulations with Physically Consistent Deep Learning
- [Techniques for deep learning on satellite and aerial imagery](https://github.com/satellite-image-deep-learning/techniques)
- [MetNet](https://github.com/openclimatefix/metnet): PyTorch Implementation of Google Research's [MetNet](https://arxiv.org/abs/2003.12140) for short term weather forecasting
- [ClimateLearn](https://github.com/aditya-grover/climate-learn): Python library for accessing state-of-the-art climate data and machine learning models in a standardized, straightforward way. This library provides access to multiple datasets, a zoo of baseline approaches, and a suite of metrics and visualizations for large-scale benchmarking of statistical downscaling and temporal forecasting methods.
- [RainNet](https://github.com/hydrogo/rainnet): a convolutional neural network for radar-based precipitation nowcasting
- [SmaAt-UNet](https://github.com/HansBambel/SmaAt-UNet): Precipitation Nowcasting using a Small, Attentive UNet-Architecture
- [ClimaX](https://github.com/microsoft/ClimaX): Foundation model for weather & climate, developed by Microsoft
- [Stiff-PINN](https://github.com/DENG-MIT/Stiff-PINN): Stiff-PINN: Physics-Informed Neural Network for Stiff Chemical Kinetics

# Google Earth Engine
- [wxee](https://github.com/aazuspan/wxee): A Python interface between Earth Engine and xarray for processing time series data
- [geemap](https://github.com/gee-community/geemap): A Python package for interactive mapping with Google Earth Engine, ipyleaflet, and ipywidgets.

# Working with Image data
- [Pillow](https://github.com/python-pillow/Pillow): image processing
- [scikit-image](https://github.com/scikit-image/scikit-image): Image processing in Python
- [opencv](https://github.com/opencv/opencv): The most popular Open Source Computer Vision Library in Python
- [imageio](https://github.com/imageio/imageio): Python library for reading and writing image data
- [instafilter](https://github.com/thoppe/instafilter): Modifiy images using Instagram-like filters in python

