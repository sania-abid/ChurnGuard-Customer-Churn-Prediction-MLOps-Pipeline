FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install ".[mlops]"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "churn_app.api:app", "--host", "0.0.0.0", "--port", "8000"]
