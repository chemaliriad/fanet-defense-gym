PY ?= python

.PHONY: install lint test test-all repro figures docker k8s-render

install:
	$(PY) -m pip install -e ".[dev,train,viz]"

lint:
	ruff check .
	ruff format --check .
	mypy

test:
	pytest -m "not slow"

test-all:
	pytest

# Full protocol (about an hour on a 4-core laptop): tune, train 5 seeds x 3 algorithms, evaluate.
repro:
	$(PY) scripts/run_experiments.py --steps 300000 --seeds 0 1 2 3 4
	$(PY) scripts/make_figures.py

figures:
	$(PY) scripts/make_figures.py

docker:
	docker build -t fanet-defense:0.1.0 .

k8s-render:
	kubectl kustomize k8s/base
