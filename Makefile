## CONFIG
PYTHONPATH := $(PYTHONPATH):$(pwd)
SHELL := /bin/bash -v

## VARIABLES
VIRTUAL_ENV_NAME := $(shell poetry env info -p | rev | cut -d'/' -f1 | rev)

## COMMANDS
kernel-install:
	@poetry run python -m ipykernel install --user --name $(VIRTUAL_ENV_NAME)

install:
	@poetry install
	@make kernel-install

cleanup:
	@rm .coverage
	@rm -rf **/__pycache__
	@rm -rf ~/Library/Jupyter/kernels/$(VIRTUAL_ENV_NAME)
	@poetry env remove python

shell:
	@poetry shell

run:
	@poetry run python src/main.py

watch:
	@watchman-make -p 'src/**/*.py' -r 'make run'

test:
	@poetry run pytest src/__tests__/ -vv -s

test-coverage:
	@poetry run pytest src/__tests__/ -vv -s --cov=src --cov-report=term-missing

test-watch:
	@watchman-make -p 'src/**/*.py' -r 'make test'

test-coverage-watch:
	@watchman-make -p 'src/**/*.py' -r 'make test-coverage'

notebook:
	@poetry run jupyter notebook
