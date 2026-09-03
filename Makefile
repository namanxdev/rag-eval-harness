# Windows users without make: run the commands directly, they are one line each.
.PHONY: test eval install

install:
	pip install -r requirements-dev.txt

test:
	pytest

eval:
	python run_eval.py
