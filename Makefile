.PHONY: install install-dev run test lint format clean

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# ── Run ────────────────────────────────────────────────────────────────────────

run:
	streamlit run app.py

# Run the full crew pipeline directly (bypasses Streamlit)
run-crew:
	python crew.py

# ── Test ───────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

# ── Code Quality ───────────────────────────────────────────────────────────────

lint:
	flake8 agents/ tasks/ utils/ config/ app.py crew.py --max-line-length=120

format:
	black agents/ tasks/ utils/ config/ app.py crew.py --line-length=120

# ── Clean ──────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage logs/*.log
