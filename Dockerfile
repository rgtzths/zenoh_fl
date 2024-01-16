FROM tensorflow/tensorflow:2.15.0

# ENV USER mpiuser

ENV HOME=/root

RUN apt update && apt -y install openmpi-bin openssh-server libopenmpi-dev wget 
RUN apt remove --purge python3.11* -y
RUN apt install -y python3.10-dev python3.10-distutils python3.10-venv
RUN mkdir /var/run/sshd

RUN cd /tmp && wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py

RUN pip3 install mpi4py scikit-learn

WORKDIR ${HOME}

RUN ssh-keygen -t rsa -f ${HOME}/.ssh/id_rsa

WORKDIR ${HOME}/.ssh/
RUN cat id_rsa.pub >> authorized_keys
RUN echo "StrictHostKeyChecking accept-new" > config
RUN echo $(ls)

WORKDIR ${HOME}

COPY . .

RUN pip3 install -r ZENOH/requirements.txt

RUN chmod +x ${HOME}/federated_learning.py

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
