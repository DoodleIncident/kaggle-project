all: out/output.csv

out/output.csv: out/kinds.csv out/sentiments.csv
	. venv/bin/activate; python lilo.py

out/kinds.csv: multilabel.py
	mkdir -p out/
	. venv/bin/activate; python multilabel.py

out/sentiments.csv: predict.py mdl/sentiments.save
	mkdir -p out/
	. venv/bin/activate; python predict.py

mdl/sentiments.save: train.py npy/input_tokens.npz
	mkdir -p mdl/
	. venv/bin/activate; python train.py

npy/input_tokens.npz: preprocess.py
	mkdir -p npy/
	. venv/bin/activate; python preprocess.py

clean:
	rm -rf #out/ npy/ mdl/
