.PHONY: quality style pip-solve tests

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .