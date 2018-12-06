#Instructions
1. CST reviewed statements are saved in the data folder
2. Scripts for preprocessing and the neural network are saved in the scripts folder
2. To get the word embeddings of all statements in the cst-reviewed statements file run 'python .\get_word2vec.py' in the data folder
3. In the analysis folder you can test AUC of neural network and other models. In the analysis folder run:
	a. python .\neural_network_testing.py
	b. python .\other_models_testing.py
4. To train the neural network run 'python .\train.py' in the source folder. This will output an .h5 file
5. To convert the .h5 file to a json file to be used in the solver run 'python .\get_weights.py'