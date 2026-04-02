.PHONY: setup train tune evaluate api streamlit test lint clean

setup:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

validate:
	python -m src.data.validator

train:
	python -m src.models.train

tune:
	python -m src.models.tuning

evaluate:
	python -m src.models.evaluate

explain:
	python -m src.models.explain

api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

streamlit:
	streamlit run app/streamlit_app.py

drift:
	python -m monitoring.drift_report

test:
	pytest tests/ -v

lint:
	ruff check src/ app/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache mlruns/
