include .env
PARAMS_FILE := $(strip $(params))
EXPERIMENT := $(shell cat $(PARAMS_FILE) | shyaml get-value experiment)
DATASETS := $(shell cat $(PARAMS_FILE) | shyaml get-value data_params.datasets)

all: setup train quotient

setup: 
	@echo "Generating Experiments"
	@poetry run python src/setup.py

train: 
	@echo "Beginning Model Training"
	@poetry run python src/train.py

quotient:
	@echo "Computing Topological Quotients"
	@poetry run python src/quotient.py


clean-experiment: clean-configs clean-results
	@rm -rf src/experiments/${EXPERIMENT}/

clean-configs:
	@echo "Cleaning Configs for ${EXPERIMENT}"
	@rm -rf src/experiments/${EXPERIMENT}/configs

clean-results:
	@echo "Cleaning Results for ${EXPERIMENT}"
	@rm -rf src/experiments/${EXPERIMENT}/results

