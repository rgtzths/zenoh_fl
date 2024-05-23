//
// Copyright (c) 2024 Gabriele Baldoni
//
// Contributors:
//   Gabriele baldoni, <gabriele@zettascale.tech>
//

use std::{
    collections::{HashMap, VecDeque},
    ops::Deref,
    sync::{Arc, Condvar},
};

use bitcode::{Decode, Encode};
use flume::Receiver;
use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods,
    types::{PyDict, PyInt, PyString},
    IntoPy, Py, PyAny, PyResult, Python, ToPyObject,
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

pub const ALL_TAG: i8 = -1;
pub const ANY_TAG: i8 = -2;

#[pyclass]
#[derive(Encode, Decode, Debug)]
pub struct ZCommData {
    pub src: i8,
    pub dest: i8,
    pub tag: i8,
    pub data: Arc<Vec<u8>>,
}

#[derive(Clone)]
pub struct ZComm {
    pub session: Arc<Session>,
    pub rank: i8,
    pub data: Arc<RwLock<HashMap<i8, HashMap<i8, VecDeque<ZCommData>>>>>,
    pub cv_map: Arc<RwLock<HashMap<i8, HashMap<i8, Condvar>>>>,
    pub msg_queue: Arc<RwLock<VecDeque<(i8, i8)>>>,
    pub expected: i8,
    pub ke_data: String,
    pub ke_live_queriable: String,
    pub live_queriable: Arc<Queryable<'static, Receiver<Query>>>,
    pub data_sub: Arc<Subscriber<'static, Receiver<Sample>>>,
}

impl ZComm {
    pub async fn new(rank: i8, workers: i8, locator: String) -> Result<Self, Error> {
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

    pub async fn send(&self, dest: i8, data: Arc<Vec<u8>>, tag: i8) -> Result<(), Error> {
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

    pub async fn bcast(&self, root: i8, data: Arc<Vec<u8>>, tag: i8) -> Result<ZCommData, Error> {
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

    pub async fn recv_single(&self, src: i8, tag: i8) -> Result<ZCommData, Error> {
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

            index = queue_guard.iter().position(|(s, _t)| *s == src);
            // here we should cover the ANY_TAG case
            data = data_guard
                .get_mut(&src)
                .and_then(|hm| hm.get_mut(&tag).and_then(|dq| dq.pop_front()));

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

    pub async fn recv(&self, src: i8, tag: i8) -> Result<HashMap<i8, ZCommData>, Error> {
        if src == ALL_SRC {
            return self.recv_from_all(tag).await;
        } else if src == ANY_SRC {
            return self.recv_from_any(tag).await;
        }

        let mut data = HashMap::new();

        match tag {
            ANY_TAG => {
                let mut any_tag: Option<i8> = None;
                while any_tag.is_none() {
                    let mut data_guard = self.data.write().await;
                    any_tag = data_guard
                        .get_mut(&src)
                        .and_then(|hm| Some(hm.keys().map(|i| *i).collect::<Vec<i8>>()))
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
                    .map(|r| r.sample.and_then(|s| Ok(s.payload.contiguous().to_vec())))
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

    pub async fn recv_from_all(&self, tag: i8) -> Result<HashMap<i8, ZCommData>, Error> {
        let mut data = HashMap::new();

        for i in 0..self.expected {
            if i == self.rank {
                continue;
            }

            data.insert(i, self.recv_single(i, tag).await?);
        }

        Ok(data)
    }

    pub async fn recv_from_any(&self, tag: i8) -> Result<HashMap<i8, ZCommData>, Error> {
        let mut data = HashMap::new();

        while self.msg_queue.read().await.len() == 0 {
            yield_now().await;
        }

        let mut guard = self.msg_queue.write().await;
        // should not unwrap here but len is > 0
        let ready_src = guard.pop_front().map(|(src, _)| src).unwrap();

        data.insert(ready_src, self.recv_single(ready_src, tag).await?);

        Ok(data)
    }
}

#[pyclass]
pub struct ZCommPy {
    pub(crate) inner: Arc<ZComm>,
}

#[pymethods]
impl ZCommPy {
    #[new]
    pub fn new<'p>(
        py: Python<'p>,
        rank: &'p PyInt,
        workers: &'p PyInt,
        locator: &'p PyString,
    ) -> PyResult<Self> {
        let rank: i8 = rank.extract()?;
        let workers: i8 = workers.extract()?;
        let locator: String = locator.extract()?;

        pyo3_asyncio::tokio::run(py, async move {
            let inner = ZComm::new(rank, workers, locator)
                .await
                .map_err(|_| PyValueError::new_err("Unable to create ZComm"))?;
            let inner = Arc::new(inner);
            Ok(ZCommPy { inner })
        })
    }

    pub fn start<'p>(&'p self, _py: Python<'p>) -> PyResult<()> {
        let _ = self.inner.start();
        Ok(())
    }

    pub fn wait<'p>(&self, py: Python<'p>) -> PyResult<&'p PyAny> {
        let c_inner = self.inner.clone();

        pyo3_asyncio::tokio::future_into_py(py, async move {
            c_inner
                .wait()
                .await
                .map_err(|_| PyValueError::new_err("Cannot wait for other nodes"))?;
            Ok(Python::with_gil(|py| py.None()))
        })
    }

    pub fn send<'p>(
        &'p self,
        py: Python<'p>,
        src: &'p PyAny,
        tag: &'p PyAny,
    ) -> PyResult<&'p PyAny> {
        let inner = self.inner.clone();
        let src: i8 = src.extract()?;
        let tag: i8 = tag.extract()?;
        let mut py_dict = PyDict::new(py);

        pyo3_asyncio::tokio::future_into_py(py, async move {
            let _res = inner
                .recv(src, tag)
                .await
                .map_err(|_| PyValueError::new_err("Cannot receive data"))?;
            // _res.iter().map(|(k,v)|
            //     py_dict.set_item(k, v.into_py(py))
            // );
            // Ok(py_dict)
            Ok(Python::with_gil(|py| py.None()))
        })
    }
}
