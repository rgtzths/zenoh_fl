from zcomm import ZCommPy, ZCommDataPy, TAGS, SRCS
import asyncio
import argparse
import pickle
import logging

async def asyncio_main(rank, workers, locator):
    zcomm = await ZCommPy.new(rank,workers, locator)
    zcomm.start()

    logging.info(f'Wait for all nodes [{workers}]...')
    await zcomm.wait()

    logging.info("All nodes discovered")

    # send to all nodes

    data = await zcomm.bcast(root=0, data=b'123', tag=100)

    logging.info(f"[Rank {rank}][BCAST] Data: {data}")

    await asyncio.sleep(2)

    ## send to all nodes (p2p)

    if rank == 0:
        for i in range(1, workers+1):
            await zcomm.send(dest=i, data=pickle.dumps(123), tag=10)
            logging.info(f"[Rank {rank}][P2P] Send to {i} 10")
    else:
        data = await zcomm.recv(src=0, tag=10)
        for src, comm in data.items():
            logging.info(f"[Rank {rank}][P2P] Data {pickle.loads(comm.data)}")

    await asyncio.sleep(2)

    ## receive from all
    if rank == 0:
        for i in range(1, workers+1):
            data = await zcomm.recv(src=i, tag=20)
            logging.info(f"[Rank {rank}][P2P] Data {data}")
           
    else:
        await zcomm.send(0, b'321', 20)
        logging.info(f"[Rank {rank}][P2P] Send to 0 20")


    ## receive from all
    if rank == 0:
        data = await zcomm.recv(src=-1, tag=30)
        logging.info(f"[Rank {rank}][P2P] Data {data}")
    else:
        await zcomm.send(0, b'987', 30)
        logging.info(f"[Rank {rank}][P2P] Send to 0 30")

    await asyncio.sleep(2)

    ## receive from any
    if rank == 0:
        data = await zcomm.recv(src=-2, tag=40)
        logging.info(f"[Rank {rank}][P2P] Data {data}")
           
    else:
        await zcomm.send(0, b'456', 40)
        logging.info(f"[Rank {rank}][P2P] Send to 0 40")

    ## receive from any, any_tag
    if rank == 0:
        data = await zcomm.recv(src=-2, tag=-2)
        logging.info(f"[Rank {rank}][P2P] Data {data}")
           
    else:
        await zcomm.send(0, b'789', 50)
        logging.info(f"[Rank {rank}][P2P] Send to 0 50")

    await asyncio.sleep(2)
    return None


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help=f"rank", type=int)
    parser.add_argument("-w", help="number of workers", type=int)
    parser.add_argument("-l", help="zenoh locator", default="tcp/127.0.0.1:7447", type=str)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.DEBUG)

    asyncio.run(asyncio_main(args.r, args.w, args.l))