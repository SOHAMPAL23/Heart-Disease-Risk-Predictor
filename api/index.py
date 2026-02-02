# Vercel-compatible API route
from backend.app import app as api_app

# This creates the required export for Vercel
app = api_app

# The application is available as both 'app' and 'api_app'
application = api_app