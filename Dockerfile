FROM tiangolo/uvicorn-gunicorn:python3.8

RUN pip install --upgrade pip

RUN adduser -disabled-password myuser
USER myuser
WORKDIR /home/myuser

COPY --chown=myuser:myuser requirements.txt ./requirements.txt
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --user -r ./requirements.txt
RUN pip install ray[tune] --user

ENV PATH="/home/myuser/.local/bin:${PATH}"

# RUN tensorboard --logdir ~/ray_results/ &> /dev/null &
COPY --chown=myuser:myuser . .
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7777"]
CMD ["tensorboard", "--logdir", "~/ray_results/", "--host", "0.0.0.0"]