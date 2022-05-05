.PHONY: style quality test test-cov

check_dirs := package/ scripts/ tests/
test_dirs := package/

style:
	black $(check_dirs)
	isort $(check_dirs)
	flake8 $(check_dirs)

quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

test:
	pytest

test-cov:
	pytest --cov-branch --cov $(test_dirs)
