PY ?= python
VENV ?= venv

.PHONY: venv install test demo run lint format

venv:
	$(PY) -m venv $(VENV)
	$(VENV)/Scripts/python -m pip install -U pip
	$(VENV)/Scripts/python -m pip install -r requirements.txt

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q -W ignore::UserWarning

demo:
	$(PY) diffusionforge/examples/run_demo.py

run:
	$(PY) -m diffusionforge.cli run

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .
