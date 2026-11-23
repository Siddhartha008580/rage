.PHONY: test benchmark-baseline benchmark-hybrid bench-diff

test:
	pytest -q tests/

benchmark-baseline:
	@echo "No benchmark fixtures are committed; baseline benchmark is not yet runnable"

benchmark-hybrid:
	@echo "No benchmark fixtures are committed; hybrid benchmark is not yet runnable"

bench-diff:
	@echo "No benchmark results are committed; benchmark diff is not yet runnable"
