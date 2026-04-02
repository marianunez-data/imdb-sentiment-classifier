FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.135.2 \
    uvicorn==0.42.0 \
    joblib==1.5.3 \
    scikit-learn==1.8.0 \
    lightgbm==4.6.0 \
    shap==0.51.0 \
    numpy==2.4.4 \
    pandas==2.3.3 \
    pydantic==2.12.5 \
    pyyaml==6.0.3 \
    structlog==25.5.0 \
    mangum==0.21.0 \
    spacy==3.8.4

RUN python -m spacy download en_core_web_sm

COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY models/lr_tuned_calibrated_pipeline.joblib models/lr_tuned_calibrated_pipeline.joblib

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
