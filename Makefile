.SILENT:
.PHONY: install-git-hooks
install-git-hooks:
	poetry run pre-commit install -f --hook-type pre-commit --hook-type pre-push >/dev/null 2>&1

.PHONY: lint-src
lint-src:
	poetry run isort src --check
	poetry run black src --check
	poetry run pylint src
	poetry run flake8 src
	poetry run mypy --explicit-package-bases src

.PHONY: lint-test
lint-test:
	poetry run isort tests --check
	poetry run black tests --check
	poetry run pylint tests --ignore=tests/conftest.py
	poetry run flake8 tests
	poetry run mypy --explicit-package-bases tests

.PHONY: lint
lint: lint-src lint-test

.PHONY: format
format:
	poetry run autoflake --recursive src
	poetry run autoflake --recursive tests
	poetry run isort src
	poetry run isort tests
	poetry run black src
	poetry run black tests

.PHONY: test
test: 
	poetry run pytest --pyargs -rfExX

.PHONY: code-coverage
code-coverage:
	poetry run pytest --pyargs --cov-report html:build/coverage --cov-report term-missing --cov=src
