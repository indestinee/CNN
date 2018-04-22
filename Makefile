all:
	@

train:
	python3 train.py

board:
	tensorboard --logdir=./train_log

clean:
	rm -rf train_log

install:
	pip3 install numpy
	pip3 install tensorboard
	pip3 install tensorflow
	#	pip3 install tensorflow-gpu # if u need gpu version
	pip3 install progressbar2

cache:
	rm -rf cache
