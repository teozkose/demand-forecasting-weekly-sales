PY ?= python
export PYTHONPATH := src

.PHONY: data panel run report notebook test lint all

data:
	$(PY) -m demand.pipeline data

panel: data
	$(PY) -m demand.pipeline panel

run: panel
	$(PY) -m demand.pipeline run

report:
	$(PY) -m demand.pipeline report

notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_analysis.ipynb --ExecutePreprocessor.timeout=600

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

all: run report notebook
