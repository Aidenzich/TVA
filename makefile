ROOT=$(shell pwd)
SHELL=/bin/zsh
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate tva

run:
	docker compose up -d

brun:
	docker compose up --build -d

exec:
	docker exec -it azrecsys env TERM=xterm-256color script -q -c "/bin/bash" /dev/null

stop:
	docker stop azrecsys

preprocess:
	@echo "preprocess"
	@echo "ROOT: $(ROOT)"

train:
	$(CONDA_ACTIVATE); python -m src.cli.train_cli

pp:
	python -m src.cli.preprocess_cli

infer:
	python -m src.cli.inference_cli

predict:
	python -m src.cli.predict_cli

otto_infer:
	python -m src.cli.otto_inference_cli

cds:
	python -m src.cli.cornac_cli

clean:
	docker rm -f azrecsys

cleanlogs:
	rm -rf logs/*

cleanall:
	@echo "Clean logs, data, and configs"
	rm -rf logs/*
	rm -rf data/cache/dataclass/*
	rm -rf data/cache/nsample/*
	rm -rf configs/*
	rm -rf out/*

panel:
	$(CONDA_ACTIVATE); python3 -m tensorboard.main --logdir=logs