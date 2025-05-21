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

2. Build the docs
3. 
3. 
3. 

# Building wheel

In a shell, navigate to the root directory of this repository. Create a virtual environment, install the 
packages listed in `requirements.txt` and activate the virtual environment. Then run 

    python -m build --wheel --outdir artefacts

The Python wheel is stored in the directory `artefacts`. Execute `make clean`, to delete the
generated binary files in the building process of the wheel.