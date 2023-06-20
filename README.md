<img src="content/assets/header.gif" />
<hr>
<p align="center">
<b style="font-size:30vw;">Introduction to Kokkos and its application to N-BODY simulations</b>
</p>
<hr>

In this repository, you will find the code for our project for the course of *New Methods in Computational Mechanics and Physics*. It contains:

- A C++ code implementing a N-BODY simulator;
- A C++ code using **Kokkos** implementing a N-BODY simulator;
- A python code allowing to generate datasets as .txt files
- A python code to create *gif* from the results obtained.

<hr>
<p  style="font-size:20px; font-weight:bold;" align="center">
Installation & Overview
</p>
<hr>

Ocean and climate models attempt to simulate continuous processes, but are discrete and run at finite resolution. The error incurred by discretization on a finite grid, however, can be approximated by _subgrid parameterizations_ and corrected at every timestep. Subgrid parameterizations are attempting to capture the effects of scales that are not resolved on the finite grid of the climate or ocean models we are using. Subgrid parameterizations can be formulated and derived in many ways, e.g. as equations derived by physical analysis, as a neural network learned from data, or as equations again but learned from data with symbolic regression.

<hr>
<p  style="font-size:20px; font-weight:bold;" align="center">
Installation
</p>
<hr>

1. Clone the repository

2. Create an appropriate **Conda** environnement:

```
conda env create -f environment.yml
```

3. Activate the  **Conda** environnement:

```
conda activate TFE
```

4. Install locally as a package:

```
pip install --editable .
```

5. Run the code easily using the notebook [`TFE.ipynb`](./notebooks/TFE.ipynb).
