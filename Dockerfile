FROM python:3.12-slim

WORKDIR /app

COPY tools/requirements.txt tools/requirements.txt
RUN pip install --no-cache-dir -r tools/requirements.txt

COPY tools tools

ENTRYPOINT ["python"]
