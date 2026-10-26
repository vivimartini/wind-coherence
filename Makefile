PY ?= python

.PHONY: install test lint demo dev reproduce

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check gbwind tests

demo:
	$(PY) -m gbwind demo

dev:
	$(PY) -m gbwind run --period dev

reproduce:
	$(PY) -m gbwind reproduce --period dev
