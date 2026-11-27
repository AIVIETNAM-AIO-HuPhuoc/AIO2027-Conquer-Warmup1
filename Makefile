.PHONY: install setup-from-kernel setup-from-raw track-data run-charts streamlit

PYTHON ?= python3

# 1. Install project dependencies
install:
	$(PYTHON) -m pip install -r requirements.txt

# 2A. Run setup_data.py from Kaggle kernel (juncuong/warnup) and split into small CSVs
setup-from-kernel:
	$(PYTHON) setup_data.py --from-kernel

# 2B. Run setup_data.py from local raw CSV files placed in data/raw/
setup-from-raw:
	$(PYTHON) setup_data.py --from-raw

# 3. Track the generated pre-processed CSV files in Git (skips if already tracked unless FORCE=1 is set)
track-data:
	@if [ "$(FORCE)" != "1" ] && [ -f data/processed/pre-processed.csv ] && git ls-files --error-unmatch data/processed/pre-processed.csv >/dev/null 2>&1; then \
		echo "[Git Track] Pre-processed CSV files in data/processed/ already exist and are tracked. Skipping (run 'make track-data FORCE=1' or use --force to re-stage)."; \
	elif [ ! -f data/processed/pre-processed.csv ]; then \
		echo "[Git Track Error] No pre-processed CSV files found in data/processed/. Run 'make setup-from-kernel' or 'make setup-from-raw' first."; \
		exit 1; \
	else \
		git add data/processed/*.csv data/processed/purchases_by_month/*.csv; \
		git status; \
	fi

# 4. Run main.py core RFM logic, export summary CSVs/PNGs, and display interactive charts
run-charts:
	$(PYTHON) main.py --show-plots

# 5. Launch interactive Streamlit dashboard (displays all charts in web UI)
streamlit:
	streamlit run main.py
