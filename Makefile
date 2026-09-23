UV= uv
PYTHON = $(UV) run --active python3

all: install run


install:
	$(UV) sync --active

run:
	$(PYTHON) -m src

lint:
	@$(PYTHON) -m flake8 src
	@$(PYTHON) -m mypy src --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

debug:
	@$(PYTHON) -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -fr data/output
	rm -fr .mypy_cache
	rm -fr .venv
