# Pointer Sequencer ML Repository

This repository contains a package for analyzing pointer networks on 
sequencing problems. It is still a work-in-progress, I'm in the exploratory
stage of the project so if you happen to find this, please mind my unfinished
work. 

I developed the repository to accomplish two main goals: 
1. Study how transformer based neural networks trained with reinforcement 
   learning can solve complex sequencing tasks. 
2. Teach myself about deep reinforcement learning tools and standard coding
   practices. 

## Requirements

This repository requires several packages that are available for download via
the standard methods, including conda or pip. First, clone this repository to 
your computer. Then, in a command window, change directory to wherever you 
cloned the repository and use the `environment.yml` file to create a new conda 
environment. Note: I highly recommend using mamba instead of conda. The best 
way to do that is with 
[miniforge](https://github.com/conda-forge/miniforge#mambaforge) but if you 
want to use an existing conda setup then instructions are 
[here](https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install).
If you are using conda instead of mamba, replace `mamba` with `conda`, they 
work identically (except mamba is faster!).

```
cd /path/to/cloned/repository
mamba env create -f environment.yml
```

Note: I have tested and developed this code on a Windows 10 machine so cannot 
guarantee that it works on other operating systems. I think most compatibility
issues will relate to pytorch and nvidia tools, so if the environment creation 
fails, I would recommend commenting out the lines in the `environment.yml` 
file related to pytorch, (`pytorch`, `torchvision`, `torchaudio`, 
`pytorch-cuda=12.1`), creating the environment as above, then installing the 
torch packages as recommended from the  
[pytorch website](https://pytorch.org/get-started/locally/). Note that for you
to use your GPU (if it's installed), the pytorch-cuda version needs to be the 
same as whatever is installed on your computer. To figure this out, open a 
command prompt and type `nvidia-smi`. It'll show you the CUDA Version in the 
top right if it's installed. 

```
mamba create -n pointersequencer
mamba activate pointersequencer
pip install <package_name> # go in order through the environment.yml file, ignore the pytorch packages

# use whatever line of code is suggested from the pytorch website:
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Documentation
Until I learn how to build a project page, the presentation and documentation
of this repository is going to live on markdown files in the [docs](docs) 
folder. These files explain how to use this repository and present analyses of
the agents I have developed. This is a list of them with links to the file.

### Documentation for Pointer Network Experiments
1. [Toy Problem (& intro to pointer networks)](docs/part1-demonstration.md)
2. [Novel Architecture Comparison on Toy Problem](docs/part2-new_architectures.md)
3. [Tests on the Traveling Salesman Problem](docs/part3-traveling_salesman.md)
4. [A Novel Complex Sequencing Problem](docs/part4-dominoe_sequencer.md)

## Contributing
Feel free to contribute to this project by opening issues or submitting pull 
requests. I'm doing this to learn about RL and ML so suggestions, 
improvements, and collaborations are more than welcome!

## License
This project is licensed under the MIT License.
