FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install uvicorn and gunicorn
RUN pip install "uvicorn[standard]" gunicorn

COPY . .

EXPOSE 8002

CMD ["gunicorn", "backend.app:app", "--config", "gunicorn.conf.py"]