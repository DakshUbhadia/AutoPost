.DEFAULT_GOAL := help
.PHONY: help install selfcheck test test-all prereg-verify spike data hidden phase1 gonogo run report

help:
	@echo install selfcheck test test-all prereg-verify spike data hidden phase1 gonogo run report

install:
	python -m pip install -e '.[dev]' -r requirements.lock

selfcheck:
	python scripts/selfcheck.py

test:
	python -m pytest -q -m 'not slow'

test-all:
	python -m pytest -q

prereg-verify:
	python -m autopost.cli prereg verify

spike data hidden phase1 gonogo run report:
	@echo not implemented yet (see roadmap)
	@exit 1
