all: train

train: example.py npy/
	. venv/bin/activate; python example.py

npy/: build_layers.py pre/
	mkdir -p npy/
	. venv/bin/activate; python build_layers.py

pre/: preprocess.py train.csv
	mkdir -p pre/
	. venv/bin/activate; python preprocess.py

clean:
	rm -rf pre/ npy/
