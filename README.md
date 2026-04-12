# LOB-simulator


How to Run It
1. Activate your venv
```
source .venv/bin/activate
```
2. Install the project dependencies
```
pip install -e env/
pip install sortedcontainers
```
3. Run the FastAPI server locally
```
cd env
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```