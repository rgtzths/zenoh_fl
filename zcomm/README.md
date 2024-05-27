# ZCOMM rust + python

Implementation of MPI-like APIs on top of Zenoh.

### Requirements

- Rust: see the [installation page](https://www.rust-lang.org/tools/install)
- a matching version of libpython. On linux systems, it's typically packaged separately as ``libpython3.x-dev` or `python3.x-dev`.
- Python >= 3.7
- pip >= 22
- virtualenv

### How to build (release)

Create and activate a python virtual environment:

```bash
$ python3 -m virtualenv venv
$ source venv/bin/activate
```

Build the Python Wheel **within** a Python virtual environment.

```bash
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ git clone git@github.com:rgtzths/zenoh_fl.git
(venv) $ cd zenoh_fl/zcomm
(venv) $ pip3 install -r requirements-dev.txt
(venv) $ maturin build --release
```


Install the python bindings.

```bash
$ pip3 install ./target/wheels/<there should only be one .whl file here>
```

### How to build (debug)

Create and activate a python virtual environment:

```bash
$ python3 -m virtualenv venv
$ source venv/bin/activate
```

Build and install the Python Wheel **within** a Python virtual environment.

```bash
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ git clone git@github.com:rgtzths/zenoh_fl.git
(venv) $ cd zenoh_fl/zcomm
(venv) $ pip3 install -r requirements-dev.txt
(venv) $ maturin develop
```

### How to use

Once the package is installed (both debug or release), you can import the modules like this:

```python
from zcomm import ZCommPy, ZCommDataPy, TAGS, SRCS
import asyncio

async def asyncio_main():
    zcomm = await ZCommPy.new(0,1, "tcp/127.0.0.1:7447")
    zcomm.start()

    await zcomm.send(1, b'123', 1)
    return None

asyncio.run(asyncio_main())
```

See also `example.py` for a more complete example.