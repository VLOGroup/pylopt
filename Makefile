.PHONY: clean build install test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info .pytest_cache

build: 
	python -m build --outdir artefacts

install: clean
	python setup.py install

test:
	pytest -v -s tests