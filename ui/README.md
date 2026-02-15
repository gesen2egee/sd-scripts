# ANIMA UI

## Start

From repo root:

```bat
start.bat
```

This opens the local web UI at `http://127.0.0.1:8765`.

## Notes

- Existing `sd-scripts` code is not modified by this UI.
- UI backend runs with root `venv` (`venv\Scripts\python.exe`).
- Training is launched with root `venv\Scripts\accelerate.exe` and `anima_train_network.py`.
- Runtime files are written to `ui/runtime/`.
- The backend is stdlib-based and does not require extra package installation.
