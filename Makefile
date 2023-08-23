all: generate
	@echo "Training Models"
	@poetry run python src/main.py

generate: 
	@echo "Generating Experiments"
	@poetry run python src/generate_experiments.py

analyze:
	@echo "Computing Persistent Homology"