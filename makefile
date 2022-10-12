ROOT=$(shell pwd)

run:
	docker compose up -d

brun:
	docker compose up --build -d

exec:
	docker exec -it azrecsys bash

clean:
	docker rm -f azrecsys

stop:
	docker stop azrecsys

preprocess:
	@echo "preprocess"
	@echo "ROOT: $(ROOT)"

train:
	python -m src.cli.train_cli

pp:
	python -m src.cli.preprocess_cli

infer:
	python -m src.cli.inference_cli

clearlogs:
	rm -rf logs/*