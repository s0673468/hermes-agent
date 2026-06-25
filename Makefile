PYTHON ?= python3
RUFF ?= $(shell if command -v ruff >/dev/null 2>&1; then printf "ruff"; elif command -v uvx >/dev/null 2>&1; then printf "uvx ruff"; else printf "$(PYTHON) -m ruff"; fi)

.PHONY: check lint test

check: lint test

lint:
	$(RUFF) check --select E9,F63,F7,F82 agent tools hermes_cli gateway tui_gateway cron acp_adapter plugins tests

test:
	scripts/run_tests.sh
