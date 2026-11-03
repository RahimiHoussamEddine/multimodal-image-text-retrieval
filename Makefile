PYTHON ?= .venv/bin/python
PIP_CACHE_DIR := $(CURDIR)/.cache/pip
PIP_TMP_DIR := $(CURDIR)/.cache/tmp
export HF_HOME := $(CURDIR)/.cache/huggingface
export OMP_NUM_THREADS := 2
export OPENBLAS_NUM_THREADS := 2
export MKL_NUM_THREADS := 2
export TOKENIZERS_PARALLELISM := false

.PHONY: install-dev install-lock fetch validate-data cache-model encode evaluate test quality smoke-app app

install-dev:
	mkdir -p $(PIP_TMP_DIR)
	TMPDIR=$(PIP_TMP_DIR) PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu

install-lock:
	mkdir -p $(PIP_TMP_DIR)
	TMPDIR=$(PIP_TMP_DIR) PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install \
		-r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
	TMPDIR=$(PIP_TMP_DIR) PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip install --no-deps --no-build-isolation -e .

fetch:
	$(PYTHON) -m multimodal_retrieval.data fetch

validate-data:
	$(PYTHON) -m multimodal_retrieval.data validate-only

cache-model:
	$(PYTHON) -m multimodal_retrieval.embeddings --cache-model-only

encode:
	$(PYTHON) -m multimodal_retrieval.embeddings

evaluate:
	$(PYTHON) -m multimodal_retrieval.evaluate

test:
	$(PYTHON) -m pytest

quality:
	TMPDIR=$(PIP_TMP_DIR) PIP_CACHE_DIR=$(PIP_CACHE_DIR) $(PYTHON) -m pip check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

smoke-app:
	$(PYTHON) scripts/smoke_app.py

app:
	$(PYTHON) -m streamlit run app.py --server.address=127.0.0.1 --server.headless=true
