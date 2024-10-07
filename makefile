# This line forces make to rebuild these even if there are files/folders
# Whose names clash with the names of the build targets
.PHONY: build

# This is the default build target
build: triangular-simulation rectangular-simulation

triangular-simulation:
	PYTHONPATH=${PYTHONPATH}:${PWD} && jupyter nbconvert \
		--ExecutePreprocessor.timeout=0 \
		--to notebook \
		--inplace \
		--y \
		--execute simulations/competitor-comparison-triangular.ipynb
rectangular-simulation:
	PYTHONPATH=${PYTHONPATH}:${PWD} && jupyter nbconvert \
		--ExecutePreprocessor.timeout=0 \
		--to notebook \
		--inplace \
		--y \
		--execute simulations/competitor-comparison-rectangular.ipynb
