FROM tiangolo/uvicorn-gunicorn:python3.8

RUN pip install --upgrade pip

# RUN adduser -disabled-password myuser
RUN apt-get update && apt-get -y install sudo
RUN apt-get update && apt-get -y install vim
WORKDIR /home/app

COPY requirements.txt ./requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install  -r ./requirements.txt
RUN pip install ray[tune] 

ENV PATH="/home/app/.local/bin:${PATH}"

# RUN tensorboard --logdir ~/ray_results/ &> /dev/null &
COPY . .

CMD ["tensorboard", "--logdir", "~/ray_results/", "--host", "0.0.0.0"]