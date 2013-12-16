all: predict

predict: predict.py mdl/sentiments.save
	mkdir -p out/
	. venv/bin/activate; python predict.py

mdl/sentiments.save: train.py npy/input_tokens.npz
	mkdir -p mdl/
	. venv/bin/activate; python train.py

npy/input_tokens.npz: preprocess.py
	mkdir -p npy/
	. venv/bin/activate; python preprocess.py

clean:
	rm -rf pre/ npy/
