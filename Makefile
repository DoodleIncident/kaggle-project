all: train

train: npy/
	# dummy command, of course
	ls npy

npy/: build_layers.py pre/
	mkdir npy/
	. venv/bin/activate; python build_layers.py

pre/: preprocess.py train.csv
	mkdir pre/
	. venv/bin/activate; python preprocess.py

clean:
	rm -rf pre/ npy/
