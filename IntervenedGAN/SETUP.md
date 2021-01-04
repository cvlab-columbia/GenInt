## Setup
1. Install anaconda or miniconda
2. Create environment: `conda create -n ganspace python=3.7`
3. Activate environment: `conda activate ganspace`
4. Install dependencies: `conda env update -f environment.yml --prune`
5. Setup submodules: `git submodule update --init --recursive`
6. Run command `python -c "import nltk; nltk.download('wordnet')"`


#### Linux
1. Install CUDA toolkit (match the version in environment.yml)
2. Download pycuda sources from: https://pypi.org/project/pycuda/#files
3. Extract files: `tar -xzf pycuda-VERSION.tar.gz`
4. Configure: `python configure.py --cuda-enable-gl --cuda-root=/path/to/cuda`
5. Compile and install: `make install`
6. Install Glumpy: `pip install setuptools cython glumpy`

