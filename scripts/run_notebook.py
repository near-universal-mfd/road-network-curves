"""Execute a notebook from notebooks/ with the current Python; the exit code is 1 if a cell fails.
Usage: python run_notebook.py NOTEBOOK.ipynb [--save]
"""
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from paths import NOTEBOOKS

path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": NOTEBOOKS}})
t0 = time.time()
status = 0
try:
    client.execute()
except CellExecutionError as exc:
    print(f"[FAILED] {str(exc)[-3000:]}")
    status = 1
finally:
    if "--save" in sys.argv:
        nbformat.write(nb, path)
print(f"[DONE] {path} in {(time.time() - t0) / 60:.1f} min, status {status}")
sys.exit(status)
