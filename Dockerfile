FROM tensorflow/tensorflow

ENV USER mpiuser

ENV HOME=/home/${USER} 

RUN apt update && apt -y install openmpi-bin && apt -y install openssh-server

RUN mkdir /var/run/sshd

RUN adduser ${USER}

RUN usermod -aG sudo ${USER}

USER mpiuser

RUN python3 -m pip install mpi4py && python3 -m pip install scikit-learn

WORKDIR ${HOME}/.ssh/

RUN ssh-keygen -t rsa -f id_rsa

RUN cat id_rsa.pub >> authorized_keys
RUN echo $(ls)

COPY dataset/one_hot_encoding ${HOME}/code/dataset
COPY mpi_centralized_assync.py ${HOME}/code/mpi_centralized_assync.py
COPY mpi_centralized_sync.py ${HOME}/code/mpi_centralized_sync.py
COPY mpi_decentralized_assync.py ${HOME}/code/mpi_decentralized_assync.py
COPY mpi_decentralized_sync.py ${HOME}/code/mpi_decentralized_sync.py
COPY host_file ${HOME}/code/hostfile

WORKDIR ${HOME}

USER root

RUN chown -R ${USER}:${USER} ${HOME}/code

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
