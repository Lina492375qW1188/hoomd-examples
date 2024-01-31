# hoomd-examples

Most of the simulations are compatible with HOOMD-blue version 3. The latest version of HOOMD is version 4, which needs minor changes in code.

Install HOOMD-blue, freud, fresnel, signac, gsd, coxeter and their required package in conda environment. Here is one example I used for my project:
```
conda create -n patchy
conda activate patchy
conda install mamba -c conda-forge
mamba install python=3.9 numpy matplotlib jupyterlab notebook -c conda-forge
mamba install pybind11 eigen cmake qhull pytest -c conda-forge
pip install PySide6

mamba install hoomd freud fresnel -c conda-forge
mamba install signac signac-flow -c conda-forge
mamba install gsd -c conda-forge
mamba install coxeter -c conda-forge

mamba install scikit-learn -c intel
mamba install pytorch torchvision -c pytorch
mamba install ipywidgets -c conda-forge

mamba install umap-learn -c conda-forge

conda deactivate
```
