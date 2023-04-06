FROM tiangolo/uvicorn-gunicorn:python3.8
RUN pip install --upgrade pip

# RUN adduser -disabled-password myuser
RUN apt-get update && apt-get -y install sudo && apt-get -y install vim
WORKDIR /home/app

COPY requirements.txt ./requirements.txt
RUN pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install  -r ./requirements.txt
RUN pip install ray[tune] 

ENV PATH="/home/app/.local/bin:${PATH}"

# RUN tensorboard --logdir ~/ray_results/ &> /dev/null &
COPY . .

CMD ["tensorboard", "--logdir", "./logs", "--host", "0.0.0.0"]