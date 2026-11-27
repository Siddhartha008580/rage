.PHONY: test benchmark-baseline benchmark-hybrid bench-diff

PYTHON ?= python

test:
	$(PYTHON) -m pytest -q tests/

benchmark-baseline:
	$(PYTHON) scripts/eval_retrieval.py --mode vector --index-dir index

benchmark-hybrid:
	$(PYTHON) scripts/eval_retrieval.py --mode hybrid --index-dir index

bench-diff:
	$(PYTHON) scripts/eval_answers.py --index-dir index
