.PHONY: quality stylegenerate_step

quality:
	ruff check .
	ruff format --check .

style:
	ruff format .
	ruff check --fix .