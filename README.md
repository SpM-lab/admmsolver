A fast and general ADMM solver
=======================================================================
# Installation

```
pip3 install admmsolver
```

# Implementation note
[Implementation note](https://www.overleaf.com/read/fxbjmzsvwtgh)

# How to develop using VS code + docker
1. Install `Remote Development` extension of VS code.
2. Clone the repository and open the cloned directory by VS code.
3. You will be asked if you want to reopen the directory in a container. Say yes!<br>
(The first time you start a docker container, a docker image is built. This takes a few minutes).
4. Once you login in the container, all required Python packages are pre-installed and `PYTHONPATH` is set to `src`.
5. To run all unit tests and static type checkes by `mypy`, use ```./bin/runtests.sh```.

Note:
* The cloned git repository on your host file system is mounted on the working directory in the container. Created files in the container will be persisted even after the container stops.
* The full list of what are installed in the container will be found in `.devcontainer/Dockerfile`.
* If you customize `Dockerfile` and build an image for debugging, execute the command shown below on your host machine.
```
docker build -f .devcontainer/Dockerfile .
```
