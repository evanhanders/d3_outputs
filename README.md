# d3_outputs
Averaging and output tasks for domains in dedalus' d3 framework

# Installation
To install d3_outputs on your local machine, clone this repository, then navigate into it (so that you see setup.py when you use ls), and type:

> pip3 install -e .

# Getting started
Navigate to the examples folder. Try out the example file:

>  python3 test_new_marti_conv.py

It should take a few minutes to run on a single process. Once it's finished, run

> python3 plot_slices.py test_outputs/

(Note: This step requires installation of [plotpal](https://github.com/evanhanders/plotpal)). There should be three folders of figures (test_outputs/snapshots_eq, test_outputs/snapshots_mer, test_outputs/snapshots_ortho) to look at. 
