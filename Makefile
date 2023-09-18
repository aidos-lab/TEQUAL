include .env
PARAMS_FILE := $(strip $(params))
EXPERIMENT := $(shell cat $(PARAMS_FILE) | shyaml get-value experiment)
DATASETS := $(shell cat $(PARAMS_FILE) | shyaml get-value data_params.dataset)

all: setup train quotient stability

setup: 
	@echo "Generating Experiments"
	@poetry run python src/setup.py

train: 
	@echo "Beginning Model Training"
	@cd src/scripts/ && ./trainer.sh

quotient:
	@echo "Computing Topological Quotients"
	@poetry run python src/quotient.py

stability:
	@echo "Logging Stability Scores"
	@poetry run python src/stability.py


clean-experiment: clean-configs clean-results
	@poetry shell
	@rm -rf src/experiments/${EXPERIMENT}/
clean-configs:
	@poetry shell
	@echo "Cleaning Configs for ${EXPERIMENT}"
	@rm -rf src/experiments/${EXPERIMENT}/configs

clean-results:
	@poetry shell
	@echo "Cleaning Results for ${EXPERIMENT}"
	@rm -rf src/experiments/${EXPERIMENT}/results

