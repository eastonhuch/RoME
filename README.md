# RoME

The code in this repository can be used to reproduce the simulation results in the paper "RoME: A Robust Mixed-Effects Bandit Algorithm for Optimizing Mobile Health Interventions."

# Docker Usage

The purpose of including a Docker image in this repo is to make the simulations fully reproducible across different computer architectures and operating systems.
To use the image, you will first need to download Docker Desktop from [here](https://docs.docker.com/desktop/).
Then you can build the image as follows:

`docker build -t rome --no-cache -f docker/Dockerfile .`

We produced the results using 100Gb of memory and 25 CPUs in about 3 hours.
Though, the results could likely be reproduced on more modest computational resources (a personal computer) in less than 24 hours; in this case, we recommend decreasing the number of concurrent jobs in the notebooks.
To reproduce the results shown in the paper, run the following command:

```
docker run \
      -v $(pwd):/home/jovyan \
      --platform linux/amd64 \
      rome
```

The platform flag is recommended if you are running on an M1 Mac; otherwise omit it.
If you are running in a scientific computing environment that does not support Docker, then you could convert your Docker image to a .sif file and run it via Singularity.

If you would like to modify the simulation interactively, comment out the entry point in the Dockerfile.
Then rebuild the Docker container and run the following code:

```
docker run \
      -p 8888:8888 \
      -v $(pwd):/home/jovyan \
      --platform linux/amd64 \
      rome
```

Then navigate to the localhost URL to develop in Jupyter Lab.
Because the repo is attached as a Docker volume, any changes you make will be synced between the host and container.
If rebuilding the container fails, you could alternatively use the prebuilt container at [https://hub.docker.com/r/eastonhuch/rome](https://hub.docker.com/r/eastonhuch/rome).
This version is meant to be run interactively.

# License

Most of the code contained in this repository is licensed under the GPL-3 (see LICENSE.md for full text).
The reason for this is that we use the CHOLMOD module from SuiteSparse which has a copy-left GPL-3 license.
The only files with a different license are bagging_mod.py and ensemble_mod.py, which are modified from the RiverML library.
That file is licensed under the BSD 3-clause license, which is included in both files.