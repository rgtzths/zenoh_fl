#!/usr/bin/env python3
# coding: utf-8

__author__ = 'Gabriele Baldoni'
__version__ = '0.1'
__email__ = 'gabriele@zettascale.tech'
__status__ = 'Development'


import zenoh
from zenoh import config, Value, Reliability, Sample, CongestionControl
import pickle
import json
import time
import logging
from collections import deque

ALL_SRC = -1
ANY_SRC = -2
ALL_TAG = -1
ANY_TAG = -2


class ZCommData(object):
    def __init__(self, src, dest, tag, data):
        self.src = src
        self.dest = dest
        self.tag = tag
        self.data = data

    def serialize(self):
        return pickle.dumps(self)

    def deserialize(data):
        return pickle.loads(data)
    
    def __str__(self):
        return f'SRC: {self.src} DST: {self.dest} TAG: {self.tag}'


class ZComm(object):
    def __init__(self, rank, workers, locator =  None):
        self.session = self.connect_zenoh(locator)
        self.rank = rank
        self.data = {}
        self.msg_queue = deque()

        self.ke_live_queryable = f'mpi/{rank}/status'
        self.ke_data = f'mpi/{rank}/data'
        self.expected = workers+1

        self.queryable = self.session.declare_queryable(self.ke_data, self.data_cb)
        self.live_queryable = self.session.declare_queryable(self.ke_live_queryable, self.live_cb)

    def connect_zenoh(self, locator):
        conf = zenoh.Config()
        conf.insert_json5(zenoh.config.MODE_KEY, json.dumps("peer"))
        if locator is not None:
            conf.insert_json5(zenoh.config.CONNECT_KEY, json.dumps([locator]))
        zenoh.init_logger()
        session = zenoh.open(conf)
        return session

    def close(self):
        self.queryable.undeclare()
        self.live_queryable.undeclare()
        self.session.close()


    def update_data(self, zcomdata):
        src_data = self.data.get(zcomdata.src)
        data_dict = {zcomdata.tag: deque([zcomdata.data])}
        if src_data is None:
            # No data received from this source
            # creating the inner dict    
            self.data.update({zcomdata.src:data_dict})
        else:
            # if not present check if data with this tag already exists
            tag_data = src_data.get(zcomdata.tag) 
            if tag_data is None:
                # if not add it
                src_data.update(data_dict)
            else:
                # if yes add to the queue
                tag_data.append(zcomdata.data)

        self.msg_queue.append(zcomdata.src)
        #logging.debug(f'[RANK {self.rank}] Queue: {self.msg_queue} Data: {self.data}')

    def data_cb(self, query):
        #logging.debug(f'[RANK {self.rank}] DATA CB')
        ke = f'{query.selector.key_expr}'
        data = ZCommData.deserialize(query.value.payload)
        #logging.debug(f'[RANK {self.rank}] Received on {ke} - Data: {data}')
        self.update_data(data)
        query.reply(Sample(ke, b''))

    def live_cb(self, query):
        data = f'{self.rank}-up'.encode("utf-8")
        query.reply(Sample(self.ke_live_queryable, data))


    def wait(self, expected):
        for i in range(0, expected):
            if i == self.rank:
                continue
            ke = f'mpi/{i}/status'
            data = None
            while data is None:
                replies = self.session.get(ke, zenoh.Queue())
                for reply in replies.receiver:
                    if int(reply.ok.payload.decode("utf-8").split("-")[0]) == i:
                        data = "up"

    def recv(self, source, tag):
        # logging.debug(f'[RANK: {self.rank}] Receving from {source} - Tag: {tag}')
        if source == ALL_SRC:
            return self.recv_from_all(tag)
        elif source == ANY_SRC:
            return self.recv_from_any(tag)

        data = None
        while data is None:


            
            try:
                # logging.debug(f'[RANK {self.rank}] Queue: {self.msg_queue} ')
                # logging.debug(f'[RANK {self.rank}] Data dict: {self.data} ')
                index = self.msg_queue.index(source) # This raises an exception
                # logging.debug(f'[RANK: {self.rank}] Receving from {source} - Index: {index}')
                
                # logging.debug(f'[RANK {self.rank}] After Queue: {self.msg_queue} ')
                src_data = self.data.get(source)
                
                # This mean whatever tag is ready
                if tag == ANY_TAG:
                    # looping on no tags
                    while len(src_data.keys()) == 0:
                    # logging.debug(f'[RANK {self.rank}] SRC TAGS: {src_data.keys()} ')
                    tag = list(src_data.keys())[0]
                    # call self to retrieve the data with the given tag
                    data = self.recv(source, tag)
                else:
                    tag_data = src_data.get(tag) 
                    while tag_data is None: # if the data is not there wait...
                        tag_data = src_data.get(tag)
                    # if no more data with this tag remove it.
                    if len(tag_data) == 0:
                        del src_data[tag]

                    # removing the processed message
                    del self.msg_queue[index] 
                    # get the data and end the loop
                    queue_data = tag_data.popleft()

                    if len(tag_data) == 0:
                        del src_data[tag]

                    data = {(source, tag): queue_data} 

            except Exception as e:
                # logging.error(f'[RANK {self.rank}] Exception: {e} ')
                continue
        return data

    def send(self, dest, data, tag):
        msg = ZCommData(self.rank, dest, tag, data)
        expected = 1
        acks = 0
        ke = f"mpi/{dest}/data"
        #logging.debug(f'[RANK: {self.rank}] Sending on {ke} - Data: {msg}')
        while acks < expected:
            replies = self.session.get(ke, zenoh.Queue(), value = msg.serialize())
            for reply in replies:
                try:
                    if reply.ok is not None:
                        acks = 1
                except:
                    continue
            
        #logging.debug(f'[RANK: {self.rank}] Sent on {ke}')

    def bcast(self, root, data, tag):
        if self.rank == root:
            for i in range(0, self.expected):
                if i == self.rank:
                    # do not self send
                    continue
                self.send(i, data, tag)
            return data
        else:
            recv_data = self.recv(root, tag)
            return recv_data[(root, tag)]
        
    def recv_from_all(self, tag):
        data = {}
        for i in range(0, self.expected):
            if i == self.rank:
                # do not recv from self
                continue
            data.update(self.recv(i, tag))
        return data

    def recv_from_any(self, tag):
        ready_src = None
        while len(self.msg_queue) == 0:

        ready_src = self.msg_queue[0]
        #logging.debug(f"Data is {self.data}")
        return self.recv(ready_src, tag)