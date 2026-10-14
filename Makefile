.PHONY: install test lint results clean

install:        ## editable install with dev tools
	pip install -e ".[dev,exploratory]"

test:           ## unit + reproduction tests
	pytest

lint:
	ruff check src tests

results:        ## full pipeline -> reports/
	stablecoin-risk run all

clean:          ## remove generated data
	rm -rf data/processed .pytest_cache
