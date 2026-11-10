PY := .venv/bin/python
MODEL ?= Qwen/Qwen3-1.7B
ADAPTER ?= adapters/sft_r16
LIMIT ?= 200

.PHONY: setup data test exercises gold-check baseline sft-data train eval-sft compare sweep figures

setup:
	python3.13 -m venv .venv
	$(PY) -m pip install -e ".[dev]"

data:
	$(PY) scripts/download_spider.py

test:
	$(PY) -m pytest -q

exercises:
	$(PY) -m pytest tests/exercises -v -rs

gold-check:
	$(PY) -m shrinkwrap.evaluate --gold-check --out results/gold_check_dev

baseline:
	$(PY) -m shrinkwrap.evaluate --model $(MODEL) --limit $(LIMIT) --out results/baseline_dev$(LIMIT)

sft-data:
	$(PY) scripts/prepare_sft.py

train:
	$(PY) -m mlx_lm lora --config configs/sft_lora.yaml

eval-sft:
	$(PY) -m shrinkwrap.evaluate --model $(MODEL) --adapter-path $(ADAPTER) --limit $(LIMIT) --out results/$(notdir $(ADAPTER))_dev$(LIMIT)

compare:
	$(PY) -m shrinkwrap.compare results/baseline_dev$(LIMIT).jsonl results/$(notdir $(ADAPTER))_dev$(LIMIT).jsonl

sweep:
	$(PY) scripts/quant_sweep.py --model $(MODEL) --adapter-path $(ADAPTER) --limit 300

figures:
	$(PY) scripts/make_figures.py
