ssh-copy-id worker$i

i=49
scp zenof_fl.zip worker$i:~
ssh worker$i

sudo apt update
sudo apt upgrade -y
sudo apt -y install openmpi-bin libopenmpi-dev libblas-dev python-is-python3 python3-pip python3.12-venv zip swig
unzip zenof_fl.zip
cd zenoh_fl
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
cd zcomm
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
python -m pip install -r requirements-dev.txt
maturin build --release
#python -m pip install ./target/wheels/zcomm-0.1.0-cp37-abi3-manylinux_2_34_aarch64.whl
python -m pip install ./target/wheels/zcomm-0.1.0-cp37-abi3-manylinux_2_34_x86_64.whl
