# Building Sphinx documentation

1. Setup
   
   Create a virtual Python environment, activate the environment and run

         pip install sphinx sphinx_rtd_theme

   In the root directory of the repository call `sphinx-quickstart` to create to setup the 
   basic directory structure of Sphinx documentation. In particular, a directory `docs`, containing
   the configuration script **conf.py** is generated. 
2. Create doc files

   In the root directory of the project run
      
         sphinx-apidoc -o docs bilevel_optimisation/

   This command generates .rst files containing comments extracted from the Python files for the documentation. 

3. Build the docs
   
   To generate the Sphinx documentation, navigate to the `docs` directory and run `make html`.

4. Deployment

# Building wheel

In a shell, navigate to the root directory of this repository. Create a virtual environment, install the 
packages listed in `requirements.txt` and activate the virtual environment. Then run 

    python -m build --wheel --outdir artefacts

The Python wheel is stored in the directory `artefacts`. Execute `make clean`, to delete the
generated binary files in the building process of the wheel. 

# Run Tensorboard

## Local

Activate a virtual environment with pre-installed Tensorboard (if not installed, simply run `pip install tensorboard`).
Then, in a shell type `tensorboard --logdir=<path_to_tfevents_dir>`. Then the dashboard of Tensorboard is (per 
default) available at `http://localhost:6006/`.

## Remote

On the remote server run tensorboard as above. To view the tensorboard from the local machine proceed as follows

* ssh <user>@<remote> -N -f -L localhost:16006:localhost:6006
* Open localhost:16006 in the browser of the local machine