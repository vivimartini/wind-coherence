FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md DESIGN.md LICENSE ./
COPY gbwind ./gbwind
COPY config ./config
COPY tests ./tests
COPY data/interim/portfolio.csv ./data/interim/portfolio.csv
RUN pip install --no-cache-dir --no-deps -e .

# Mount data/ and results/ as volumes to keep downloads and outputs.
ENTRYPOINT ["python", "-m", "gbwind"]
CMD ["reproduce", "--period", "dev"]
