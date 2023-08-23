all: setup train quotient
	

setup: 
	@echo "Generating Experiments"
	@poetry run python src/setup.py

train: 
	@echo "Training Models"
	@poetry run python src/train.py

quotient:
	@echo "Computing Topological Quotients"
	@poetry run python src/quotient.py