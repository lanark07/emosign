# Entry shim — keeps the project root importable as a package while
# letting users launch with `python la.py` instead of `python -m app`.
from app.main import run

if __name__ == "__main__":
    run()
