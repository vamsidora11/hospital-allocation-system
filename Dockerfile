FROM python:3.11-slim AS build

WORKDIR /build
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim AS runtime

WORKDIR /app
COPY --from=build /install /usr/local
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
