all: generate
	@echo "Training Models"
	@poetry run python src/generation/main.py

generate: 
	@echo "Generating Experiments"
	@poetry run python src/generation/generate_experiments.py

analyze:
	@echo "Computing Persistent Homology"