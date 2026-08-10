COMPOSE ?= $(shell \
	if command -v docker >/dev/null 2>&1; then \
		echo "docker compose"; \
	elif command -v podman >/dev/null 2>&1; then \
		echo "podman compose"; \
	else \
		echo "docker compose"; \
	fi)

PYTHON ?= python3

.PHONY: run run-build stop clean setup-local notebooks verify probe smoke check help

help:
	@echo "JAXCode"
	@echo ""
	@echo "  make run          Build and start JupyterLab at http://localhost:8888"
	@echo "  make stop         Stop the container"
	@echo "  make clean        Stop, remove volumes, and wipe saved progress"
	@echo ""
	@echo "  make notebooks    Regenerate all notebooks from the task definitions"
	@echo "  make verify       Run every task's reference solution against its tests"
	@echo "  make probe        Attack each test suite with wrong implementations"
	@echo "  make smoke        Execute the notebooks in a real Jupyter kernel"
	@echo "  make check        verify + probe + notebooks --check (what CI runs)"
	@echo ""
	@echo "  make setup-local  Copy notebooks into ./notebooks for local Jupyter"

run:
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) up --build -d
	@echo ""
	@echo "⚡ JAXCode is running!"
	@echo "   Open http://localhost:8888"
	@echo ""

run-build: run

stop:
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) down

clean:
	@echo "Using compose backend: $(COMPOSE)"
	$(COMPOSE) down -v
	rm -f data/progress.json

setup-local:
	@mkdir -p notebooks/_original_templates
	@cp templates/*.ipynb notebooks/_original_templates/
	@cp templates/*.ipynb notebooks/
	@cp solutions/*.ipynb notebooks/
	@echo "✅ Local notebooks ready in ./notebooks/"

notebooks:
	$(PYTHON) scripts/generate_notebooks.py

verify:
	$(PYTHON) scripts/verify_tasks.py

probe:
	$(PYTHON) scripts/probe_tests.py

smoke:
	$(PYTHON) scripts/smoke_notebooks.py
	$(PYTHON) scripts/smoke_notebooks.py --templates

check: verify probe
	$(PYTHON) scripts/generate_notebooks.py --check
