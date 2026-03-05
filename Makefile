.PHONY: install install-dev test lint format train evaluate clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-neuralgcm:
	pip install -e ".[neuralgcm]"

test:
	pytest tests/ -v

lint:
	ruff check floodrisk/ tests/

format:
	ruff format floodrisk/ tests/

train:
	python scripts/train_streamflow.py --config configs/streamflow/lstm_camels.yaml

evaluate:
	python scripts/evaluate.py --config configs/streamflow/lstm_camels.yaml

download-camels:
	python scripts/download_camels.py --output_dir data/camels

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
