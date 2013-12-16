all: train

train: train.py npy/input_tokens.npz
	. venv/bin/activate; python train.py

npy/input_tokens.npz: preprocess.py
	mkdir -p npy/
	. venv/bin/activate; python preprocess.py

clean:
	rm -rf pre/ npy/
