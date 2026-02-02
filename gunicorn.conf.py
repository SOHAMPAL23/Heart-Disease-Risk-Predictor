# Gunicorn configuration file for production deployment
bind = "0.0.0.0:8002"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 100
preload_app = True