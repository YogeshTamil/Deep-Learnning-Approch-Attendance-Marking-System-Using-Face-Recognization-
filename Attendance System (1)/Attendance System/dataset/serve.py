# serve.py
from waitress import serve
from app import app

# We will use Waitress's default threading model for concurrency on Windows.
# The 'threads' parameter controls the thread pool size. The default is 4.
print("Starting Waitress server on http://127.0.0.1:5000")
serve(app, host='127.0.0.1', port=5000, threads=8) # Increased threads for better performance