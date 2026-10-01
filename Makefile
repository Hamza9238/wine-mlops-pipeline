.PHONY: install lint test train evaluate clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100 --statistics

test:
	python -m pytest tests/ -v

train:
	python -m src.train

evaluate:
	python -m src.evaluate

clean:
	rm -rf mlruns mlartifacts .pytest_cache __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
