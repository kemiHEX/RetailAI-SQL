# Start here

## Windows
Double-click `run_windows.bat`.

## macOS / Linux
```bash
chmod +x run.sh
./run.sh
```

The first launch creates a local virtual environment and installs dependencies. The Gradio app then opens in your browser.

No API key, cloud database, or model download is required: the repository already contains the SQLite database and trained scikit-learn model artifacts.

To rebuild the dataset and retrain the models from scratch:
```bash
python scripts/build_project_data.py
```
