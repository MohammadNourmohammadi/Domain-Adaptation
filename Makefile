# FGW prototype-graph domain adaptation — common commands.
#
#   make help              list every target
#   make cora              one Cora_full run into runs/run_NNN_.../
#   make paper             all five datasets, 5 seeds each
#   make index             the results table so far
#   make replot            rebuild every figure from the saved traces
#   make clean-runs        delete all run history (asks first)
#
# Override anything on the command line:
#   make cora EPOCHS=300 SEEDS="1 2 3 4 5" ARGS="--knn_augment 8"

PYTHON  ?= python3
EPOCHS  ?= 300
SEEDS   ?= 1 2 3 4 5
OUT_DIR ?= runs
ARGS    ?=

RUN = $(PYTHON) $(1) --epochs $(EPOCHS) --seeds $(SEEDS) --out_dir $(OUT_DIR) $(ARGS)

.PHONY: help twitch citation yelp cora arxiv paper index replot \
        clean-runs clean-run clean-figures test

help:
	@echo "Targets:"
	@echo "  twitch citation yelp cora arxiv   run one dataset"
	@echo "  paper                             run all five"
	@echo "  index                             print runs/index.csv as a table"
	@echo "  replot                            rebuild figures from trace.json"
	@echo "  clean-runs                        delete ALL run folders (asks)"
	@echo "  clean-run RUN=run_003_...         delete one run folder"
	@echo "  clean-figures                     delete figures, keep logs/metrics"
	@echo ""
	@echo "Variables: EPOCHS=$(EPOCHS)  SEEDS='$(SEEDS)'  OUT_DIR=$(OUT_DIR)  ARGS='$(ARGS)'"
	@echo "Example:   make cora EPOCHS=300 SEEDS='1 2 3' ARGS='--knn_augment 8'"

# ------------------------------------------------------------------ runs
twitch:
	$(call RUN,main_fgw.py)

citation:
	$(call RUN,main_citation_fgw.py)

yelp:
	$(call RUN,main_yelp_fgw.py)

cora:
	$(call RUN,main_cora_full_fgw.py)

arxiv:
	$(call RUN,main_arxiv_fgw.py)

paper: twitch citation yelp cora arxiv
	@$(MAKE) index

# ------------------------------------------------------------- inspection
index:
	@test -f $(OUT_DIR)/index.csv \
	  && column -s, -t $(OUT_DIR)/index.csv \
	  || echo "no runs yet under $(OUT_DIR)/"

replot:
	$(PYTHON) scripts/replot.py --out_dir $(OUT_DIR)

test:
	$(PYTHON) -c "import src.plots, src.run_artifacts, src.fgw_train; print('imports ok')"
	@for r in main_fgw main_citation_fgw main_yelp_fgw main_cora_full_fgw main_arxiv_fgw; do \
	  $(PYTHON) $$r.py --help >/dev/null && echo "  ok  $$r"; \
	done

# ---------------------------------------------------------------- cleanup
# Destructive, so it says what it is about to delete and waits for a yes.
# `make clean-runs FORCE=1` skips the prompt (for scripts / CI).
clean-runs:
	@if [ ! -d $(OUT_DIR) ]; then echo "nothing to clean: $(OUT_DIR)/ does not exist"; exit 0; fi
	@echo "About to delete every run under $(OUT_DIR)/:"
	@ls -1 $(OUT_DIR) | sed 's/^/  /'
	@echo "  ($$(du -sh $(OUT_DIR) | cut -f1), including index.csv)"
	@if [ "$(FORCE)" != "1" ]; then \
	  printf "Type 'yes' to delete: "; read ans; \
	  [ "$$ans" = "yes" ] || { echo "aborted, nothing deleted"; exit 1; }; \
	fi
	@rm -rf $(OUT_DIR)
	@echo "deleted $(OUT_DIR)/ — the next run starts again at run_001"

# Delete a single run:  make clean-run RUN=run_003_citation_DBLPv7
# The id is NOT reused afterwards: allocate_run_dir takes max(existing)+1, so
# removing the newest run does free its number, but removing an older one
# leaves a gap rather than renumbering anything.
clean-run:
	@test -n "$(RUN)" || { echo "usage: make clean-run RUN=run_003_citation_DBLPv7"; exit 1; }
	@test -d $(OUT_DIR)/$(RUN) || { echo "no such run: $(OUT_DIR)/$(RUN)"; exit 1; }
	@rm -rf $(OUT_DIR)/$(RUN)
	@$(PYTHON) -c "import csv, os, sys; \
p = os.path.join('$(OUT_DIR)', 'index.csv'); \
rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []; \
keep = [r for r in rows if r['run'] != '$(RUN)']; \
w = csv.DictWriter(open(p, 'w', newline=''), fieldnames=list(rows[0])) if rows else None; \
(w.writeheader(), w.writerows(keep)) if w else None; \
print('removed $(RUN) and its index.csv row')"

# Keep the logs and numbers, drop the images (e.g. before a figure redesign —
# `make replot` regenerates them from trace.json).
clean-figures:
	@find $(OUT_DIR) -type d -name figures -exec rm -rf {} + 2>/dev/null || true
	@echo "figures deleted; run 'make replot' to rebuild them from trace.json"
