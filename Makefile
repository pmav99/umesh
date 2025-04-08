.PHONY: list docs

list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | grep -E -v -e '^[^[:alnum:]]' -e '^$@$$'

dev:
	poetry install --with dev --sync
	pre-commit install

style:
	pre-commit run ruff-format -a

lint:
	pre-commit run ruff -a

mypy:
	dmypy run umesh

pyright:
	basedpyright

test:
	python -m pytest -vlx

cov:
	coverage erase
	python -m pytest \
		--durations=10 \
		--cov=umesh \
		--cov-report term-missing

deps:
	mkdir -p requirements
	pre-commit run poetry-lock -a
	pre-commit run poetry-check -a
	pre-commit run poetry-export -a
