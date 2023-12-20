FROM tensorflow/tensorflow:2.13.0

ENV HOME=/root

RUN apt update && apt -y install openssh-server python-is-python3

RUN mkdir /var/run/sshd

RUN python3 -m pip install --upgrade pip


WORKDIR ${HOME}

RUN ssh-keygen -t rsa -f ${HOME}/.ssh/id_rsa

WORKDIR ${HOME}/.ssh/
RUN cat id_rsa.pub >> authorized_keys
RUN echo "StrictHostKeyChecking accept-new" > config
RUN echo $(ls)

COPY dataset/one_hot_encoding ${HOME}/code/dataset
COPY z_centralized_sync.py ${HOME}/code/z_centralized_sync.py
COPY requirements.txt ${HOME}/code/requirements.txt
COPY entrypoint.sh ${HOME}/entrypoint.sh

WORKDIR ${HOME}
RUN python3 -m pip install -r code/requirements.txt

RUN chmod +x ${HOME}/entrypoint.sh

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
