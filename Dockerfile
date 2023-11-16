FROM tensorflow/tensorflow

# ENV USER mpiuser

ENV HOME=/root

RUN apt update && apt -y install openmpi-bin openssh-server libopenmpi-dev python-is-python3

RUN mkdir /var/run/sshd

# RUN useradd -ms /bin/bash ${USER} 

# RUN usermod -aG sudo ${USER}

# USER ${USER}

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install mpi4py && python3 -m pip install scikit-learn

WORKDIR ${HOME}

RUN ssh-keygen -t rsa -f ${HOME}/.ssh/id_rsa

WORKDIR ${HOME}/.ssh/
RUN cat id_rsa.pub >> authorized_keys
RUN echo "StrictHostKeyChecking accept-new" > config
RUN echo $(ls)

COPY dataset/one_hot_encoding ${HOME}/code/dataset
COPY mpi_centralized_assync.py ${HOME}/code/mpi_centralized_assync.py
COPY mpi_centralized_sync.py ${HOME}/code/mpi_centralized_sync.py
COPY mpi_decentralized_assync.py ${HOME}/code/mpi_decentralized_assync.py
COPY mpi_decentralized_sync.py ${HOME}/code/mpi_decentralized_sync.py
COPY host_file ${HOME}/code/hostfile
COPY entrypoint.sh ${HOME}/entrypoint.sh

WORKDIR ${HOME}

# USER root

# RUN chown -R ${USER}:${USER} ${HOME}/code
RUN chmod +x ${HOME}/entrypoint.sh

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
