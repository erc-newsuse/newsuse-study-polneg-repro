.PHONY: help init clean clean-build clean-misc lint mypy test cov-run cov-report coverage list-deps build

help:
	@echo "init - initialize environment and version control"
	@echo "clean - clean non-persistent files"
	@echo "clean-build - remove build artifacts"
	@echo "clean-misc - remove various Python file artifacts"
	@echo "lint - run linter."
	@echo "mypy - run type checker."
	@echo "test - run unit tests"
	@echo "cov-run - run tests and calculate coverage statistics"
	@echo "cov-report - display test coverage statistics"
	@echo "coverage - run tests and display coverage statistics"
	@echo "list-deps - list explicit dependencies of the project"

packages:
	pip install -e .[dev]
	pre-commit install
	R -e 'remotes::install_version("glmmTMB", version = "1.1.10", repos = "http://cran.us.r-project.org", upgrade = "never")'

structure:
	git init
	mkdir -p data/raw
	mkdir -p data/proc
	mkdir -p data/ml
	mkdir -p data/remote
	mkdir -p scripts


dvc:
	dvc init --force
	dvc remote add  --default polneg ${PWD}/data/remote --local --force
	dvc config core.autostage true
	rm -f data/raw/*.dvc
	rm -f ml/datasets/*.dvc
	@if [ "`ls data/raw`" ]; then dvc add data/raw/*; fi
	@if [ "`ls data/proc`" ]; then dvc add ml/datasets/*; fi
	dvc commit

init: structure packages dvc

clean: clean-build clean-misc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-misc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '.benchmarks' -exec rm -rf {} +
	find . -name '.pytest-cache' -exec rm -rf {} +
	find . -name '.pytest_cache' -exec rm -rf {} +
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '.ruff_cache' -exec rm -rf {} +
