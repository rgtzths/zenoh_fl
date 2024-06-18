//
// Copyright (c) 2024 Gabriele Baldoni
//
// Contributors:
//   Gabriele baldoni, <gabriele@zettascale.tech>
//

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Condvar},
};

use bitcode::{Decode, Encode};
use flume::Receiver;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    pyclass, pymethods,
    types::{PyBytes, PyDict},
    IntoPy, Py, PyAny, PyObject, PyResult, Python, ToPyObject,
};
use tokio::{sync::RwLock, task::yield_now};
use zenoh::{
    prelude::r#async::*,
    queryable::{Query, Queryable},
    subscriber::Subscriber,
};

pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

pub const ALL_SRC: i8 = -1;
pub const ANY_SRC: i8 = -2;

pub const ALL_TAG: i32 = -1;
pub const ANY_TAG: i32 = -2;

#[derive(Encode, Decode, Debug)]
pub struct ZCommData {
    pub src: i8,
    pub dest: i8,
    pub tag: i32,
    pub data: Arc<Vec<u8>>,
}

#[derive(Clone)]
pub struct ZComm {
    pub session: Arc<Session>,
    pub rank: i8,
    pub data: Arc<RwLock<HashMap<i8, HashMap<i32, VecDeque<ZCommData>>>>>,
    pub cv_map: Arc<RwLock<HashMap<i8, HashMap<i8, Condvar>>>>,
    pub msg_queue: Arc<RwLock<VecDeque<(i8, i32)>>>,
    pub expected: i8,
    pub ke_data: String,
    pub ke_live_queriable: String,
    pub live_queriable: Arc<Queryable<'static, Receiver<Query>>>,
    pub data_sub: Arc<Subscriber<'static, Receiver<Sample>>>,
}

impl ZComm {
    pub async fn new(rank: i8, workers: i8, locator: String) -> Result<Self, Error> {
        zenoh_util::log::try_init_log_from_env();

        let mut zconfig = zenoh::config::Config::default();
        let _ = zconfig.set_mode(Some(WhatAmI::Peer));
        zconfig.connect.endpoints.push(EndPoint::try_from(locator)?);

        let session = zenoh::open(zconfig).res().await?.into_arc();

        let ke_data = format!("@mpi/{rank}/data");
        let ke_live_queriable = format!("@mpi/{rank}/status");
        // declare queriable for status
        let live_queriable = session.declare_queryable(&ke_live_queriable).res().await?;
        let data_sub = session
            .declare_subscriber(&ke_data)
            .reliable()
            .res()
            .await?;

        Ok(Self {
            session,
            rank,
            data: Arc::new(RwLock::new(HashMap::new())),
            cv_map: Arc::new(RwLock::new(HashMap::new())),
            msg_queue: Arc::new(RwLock::new(VecDeque::new())),
            expected: workers + 1,
            ke_data,
            ke_live_queriable,
            live_queriable: Arc::new(live_queriable),
            data_sub: Arc::new(data_sub),
        })
    }

    pub async fn close(self) -> Result<(), Error> {
        match Arc::into_inner(self.data_sub) {
            None => (),
            Some(q) => {
                q.undeclare().res().await?;
            }
        }
        match Arc::into_inner(self.live_queriable) {
            None => (),
            Some(q) => {
                q.undeclare().res().await?;
            }
        }

        match Arc::into_inner(self.session) {
            None => (),
            Some(q) => {
                q.close().res().await?;
            }
        }

        Ok(())
    }

    pub fn start(&self) -> Result<(), Error> {
        let c_status_queriable = self.live_queriable.clone();
        let c_rank = self.rank;
        tokio::task::spawn(async move {
            while let Ok(q) = c_status_queriable.recv_async().await {
                let ke = q.key_expr().clone();
                let _ = q.reply(Ok(Sample::new(ke, c_rank))).res().await;
            }
        });

        let c_data_sub = self.data_sub.clone();
        let c_self = self.clone();
        tokio::task::spawn(async move {
            while let Ok(s) = c_data_sub.recv_async().await {
                let data = bitcode::decode::<ZCommData>(&s.payload.contiguous().to_vec()).unwrap();
                // here look for a cond_variable to unlock or create one
                let src = data.src;
                let tag = data.tag;

                // let mut cv_guard = c_self.cv_map.write().await;
                let mut data_guard = c_self.data.write().await;
                let mut queue_guard = c_self.msg_queue.write().await;
                // let cv = cv_guard
                //     .get_mut(&src)
                //     .map(|mut hm| hm.entry(tag).or_insert(Condvar::new()));
                // this could be really be optimized with a Condvar

                let local_data = data_guard
                    .entry(src)
                    .or_insert(HashMap::new())
                    .entry(tag)
                    .or_insert(VecDeque::new());

                local_data.push_back(data);
                queue_guard.push_back((src, tag));
            }
        });

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    pub async fn send(&self, dest: i8, data: Arc<Vec<u8>>, tag: i32) -> Result<(), Error> {
        tracing::debug!("send({dest},{tag}) => len({})", data.len());
        let msg = ZCommData {
            src: self.rank,
            dest,
            tag,
            data,
        };

        let ke = format!("@mpi/{dest}/data");
        self.session
            .put(&ke, bitcode::encode(&msg))
            .congestion_control(CongestionControl::Block)
            .priority(Priority::DataHigh)
            .res()
            .await?;

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    pub async fn bcast(&self, root: i8, data: Arc<Vec<u8>>, tag: i32) -> Result<ZCommData, Error> {
        tracing::debug!("bast({root},{tag}) => len({})", data.len());
        if self.rank == root {
            for i in 0..self.expected {
                if i == self.rank {
                    continue;
                }

                // Creating the futures and using futures::join_all
                // can be a possible optimization
                self.send(i, data.clone(), tag).await?;
            }

            return Ok(ZCommData {
                src: root,
                dest: root,
                tag,
                data,
            });
        }

        self.recv_single(root, tag).await
    }

    #[tracing::instrument(skip(self))]
    pub async fn recv_single(&self, src: i8, tag: i32) -> Result<ZCommData, Error> {
        tracing::debug!("recv_single({src},{tag})");
        if src == ALL_SRC {
            panic!("The recv_single is supposed to receive from a single source");
        } else if src == ANY_SRC {
            panic!("The recv_single is supposed to receive from a specific source");
        }

        let mut data: Option<ZCommData> = None;
        let mut index: Option<usize> = None;

        while data.is_none() && index.is_none() {
            let mut data_guard = self.data.write().await;
            let mut queue_guard = self.msg_queue.write().await;

            index = queue_guard.iter().position(|(s, t)| *s == src && *t == tag);
            tracing::debug!("Position is {index:?}");
            // here we should cover the ANY_TAG case
            data = data_guard
                .get_mut(&src)
                .and_then(|hm| hm.get_mut(&tag).and_then(|dq| dq.pop_front()));
            tracing::debug!("Data is {data:?}");

            // removing from received if data is found
            match data {
                Some(_) => {
                    index.map(|i| queue_guard.remove(i));
                }
                None => (),
            }

            tokio::task::yield_now().await;
        }

        // not nice to unwrap but need to find a better way to do it
        Ok(data.unwrap())
    }

    #[tracing::instrument(skip(self))]
    pub async fn recv(&self, src: i8, tag: i32) -> Result<HashMap<i8, ZCommData>, Error> {
        tracing::debug!("recv({src},{tag})");
        if src == ALL_SRC {
            return self.recv_from_all(tag).await;
        } else if src == ANY_SRC {
            return self.recv_from_any(tag).await;
        }

        let mut data = HashMap::new();

        match tag {
            ANY_TAG => {
                let mut any_tag: Option<i32> = None;
                while any_tag.is_none() {
                    let mut data_guard = self.data.write().await;
                    any_tag = data_guard
                        .get_mut(&src)
                        .and_then(|hm| Some(hm.keys().copied().collect::<Vec<i32>>()))
                        .unwrap_or_default()
                        .pop();
                    yield_now().await;
                }
                // it is not none, but should fine a better way than unwrap
                data.insert(src, self.recv_single(src, any_tag.unwrap()).await?);
            }

            _ => {
                data.insert(src, self.recv_single(src, tag).await?);
            }
        }

        Ok(data)
    }

    #[tracing::instrument(skip(self))]
    pub async fn wait(&self) -> Result<(), Error> {
        // this could be improved with Group Management, but information about who #
        // is in the group is not available yet
        // thus busy looping
        'outer: for i in 0..self.expected {
            if i == self.rank {
                continue 'outer;
            }
            let ke = format!("@mpi/{i}/status");
            let mut data: Vec<Vec<u8>> = vec![];

            while data.is_empty() {
                data = self
                    .session
                    .get(&ke)
                    .res()
                    .await?
                    .iter()
                    .map(|r| r.sample.map(|s| s.payload.contiguous().to_vec()))
                    .collect::<Result<Vec<Vec<u8>>, _>>()
                    .unwrap_or_default();

                if let Some(data) = data.first() {
                    if let Some(b) = data.first() {
                        if (*b as i8) == i {
                            continue 'outer;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    pub async fn recv_from_all(&self, tag: i32) -> Result<HashMap<i8, ZCommData>, Error> {
        tracing::debug!("recv_from_all({tag})");
        let mut data = HashMap::new();

        for i in 0..self.expected {
            if i == self.rank {
                continue;
            }

            data.insert(i, self.recv_single(i, tag).await?);
        }

        Ok(data)
    }

    #[tracing::instrument(skip(self))]
    pub async fn recv_from_any(&self, tag: i32) -> Result<HashMap<i8, ZCommData>, Error> {
        tracing::debug!("recv_from_any({tag})");
        let mut data = HashMap::new();

        while self.msg_queue.read().await.len() == 0 {
            yield_now().await;
        }

        let (ready_src, ready_tag) = match tag {
            ANY_TAG => {
                let guard = self.msg_queue.read().await;
                // should not unwrap here but len is > 0
                let (ready_src, ready_tag) = guard.front().map(|(src, tag)| (*src, *tag)).unwrap();
                drop(guard);
                (ready_src, ready_tag)
            }
            _ => {
                let mut pos = self
                    .msg_queue
                    .read()
                    .await
                    .iter()
                    .position(|(_s, t)| *t == tag);
                while pos.is_none() {
                    yield_now().await;
                    pos = self
                        .msg_queue
                        .read()
                        .await
                        .iter()
                        .position(|(_s, t)| *t == tag);
                }

                let (ready_src, ready_tag) = self
                    .msg_queue
                    .read()
                    .await
                    .get(pos.unwrap())
                    .map(|(s, t)| (*s, *t))
                    .unwrap();
                (ready_src, ready_tag)
            }
        };

        tracing::debug!("recv_from_any({tag}) ready_src: {ready_src} ready_tag: {ready_tag}");

        data.insert(ready_src, self.recv_single(ready_src, tag).await?);

        Ok(data)
    }
}

// Note: done like this: https://github.com/wyfo/pyo3-async
fn tokio() -> &'static tokio::runtime::Runtime {
    use std::sync::OnceLock;
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| tokio::runtime::Runtime::new().unwrap())
}

#[pyclass]
pub struct ZCommPy {
    pub(crate) inner: Arc<ZComm>,
}

#[pymethods]
impl ZCommPy {
    #[staticmethod]
    pub async fn new<'p>(rank: i8, workers: i8, locator: String) -> PyResult<Self> {
        let inner = tokio()
            .spawn(ZComm::new(rank, workers, locator))
            .await
            .map_err(|_| PyValueError::new_err("Unable to create ZComm"))?
            .map_err(|_| PyValueError::new_err("Unable to create ZComm"))?;

        let inner = Arc::new(inner);
        Ok(ZCommPy { inner })
    }

    pub fn start<'p>(&'p self) -> PyResult<()> {
        let c_inner = self.inner.clone();
        tokio().spawn(async move { c_inner.start() });

        Ok(())
    }

    pub async fn wait<'p>(&self) -> PyResult<()> {
        let c_inner = self.inner.clone();
        tokio()
            .spawn(async move { c_inner.wait().await })
            .await
            .map_err(|_| PyValueError::new_err("Cannot wait for other nodes"))?
            .map_err(|_| PyValueError::new_err("Cannot wait for other nodes"))?;
        Ok(())
    }

    pub async fn recv<'p>(&'p self, src: i8, tag: i32) -> PyResult<PyObject> {
        let inner = self.inner.clone();

        let res = tokio()
            .spawn(async move { inner.recv(src, tag).await })
            .await
            .map_err(|_| PyValueError::new_err("Cannot receive data"))?
            .map_err(|_| PyValueError::new_err("Cannot receive data"))?;
        let ret = Python::with_gil(|py| into_py_dict(py, res));
        Ok(ret)
    }

    pub async fn send<'p>(&'p self, dest: i8, data: Vec<u8>, tag: i32) -> PyResult<()> {
        let inner = self.inner.clone();
        let data = Arc::new(data);
        tokio()
            .spawn(async move { inner.send(dest, data, tag).await })
            .await
            .map_err(|_| PyValueError::new_err("Cannot send data"))?
            .map_err(|_| PyValueError::new_err("Cannot receive data"))?;
        Ok(())
    }

    pub async fn bcast<'p>(&'p self, root: i8, data: Vec<u8>, tag: i32) -> PyResult<ZCommDataPy> {
        let inner = self.inner.clone();
        let data = Arc::new(data);
        let res = tokio()
            .spawn(async move { inner.bcast(root, data, tag).await })
            .await
            .map_err(|_| PyValueError::new_err("Cannot receive data"))?
            .map_err(|_| PyValueError::new_err("Cannot receive data"))?;
        let ret = Python::with_gil(|py| ZCommDataPy::from_rust(py, &res));
        Ok(ret)
    }
}

#[pyclass(subclass)]
pub struct ZCommDataPy {
    #[pyo3(get, set)]
    pub src: Py<PyAny>,
    #[pyo3(get, set)]
    pub dest: Py<PyAny>,
    #[pyo3(get, set)]
    pub tag: Py<PyAny>,
    #[pyo3(get, set)]
    pub data: Py<PyBytes>,
}

impl ZCommDataPy {
    pub fn from_rust(py: Python, value: &ZCommData) -> Self {
        let data = PyBytes::new_bound(py, value.data.as_ref()).into();

        Self {
            src: value.src.to_object(py),
            dest: value.dest.to_object(py),
            tag: value.tag.to_object(py),
            data,
        }
    }
}

#[pymethods]
impl ZCommDataPy {
    fn __repr__(&self) -> String {
        format!(
            "ZCommDataPy(src:{}, dest:{}, tag:{}, data:{})",
            self.src, self.dest, self.tag, self.data
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

pub(crate) fn into_py_dict(py: Python, data: HashMap<i8, ZCommData>) -> PyObject {
    let py_dict = PyDict::new_bound(py);

    data.iter().for_each(|(k, v)| {
        let _ = py_dict.set_item(k, ZCommDataPy::from_rust(py, v).into_py(py));
    });
    py_dict.to_object(py)
}

#[pyclass]
#[derive(Clone, Copy)]
enum TAGS {
    ALL = -1,
    ANY = -2,
}

#[pyclass]
#[derive(Clone, Copy)]
enum SRCS {
    ALL = -1,
    ANY = -2,
}

#[pymodule]
fn zcomm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ZCommDataPy>()?;
    m.add_class::<ZCommPy>()?;
    m.add_class::<TAGS>()?;
    m.add_class::<SRCS>()?;
    Ok(())
}
