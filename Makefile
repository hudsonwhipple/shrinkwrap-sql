PY := .venv/bin/python
MODEL ?= Qwen/Qwen3-1.7B
LIMIT ?= 200

.PHONY: setup data test gold-check baseline

setup:
	python3.13 -m venv .venv
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) scripts/download_spider.py

test:
	$(PY) -m pytest -q

gold-check:
	$(PY) -m shrinkwrap.evaluate --gold-check --out results/gold_check_dev

baseline:
	$(PY) -m shrinkwrap.evaluate --model $(MODEL) --limit $(LIMIT) --out results/baseline_dev$(LIMIT)
