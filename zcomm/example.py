from zcomm import ZCommPy, ZCommDataPy, TAGS, SRCS
import asyncio
import argparse



async def asyncio_main(rank, workers, locator):
    zcomm = await ZCommPy.new(rank,workers, locator)
    zcomm.start()

    print('Wait for all nodes...')
    await zcomm.wait()

    print("All nodes discovered")

    # send to all nodes

    data = await zcomm.bcast(0, b'123', 100)

    print(f"[Rank {rank}][BCAST] Data: {data}")

    await asyncio.sleep(2)

    ## send to all nodes (p2p)

    if rank == 0:
        for i in range(1, workers+1):
            await zcomm.send(i, b'123', 10)
            print(f"[Rank {rank}][P2P] Send to {i} 10")
    else:
        data = await zcomm.recv(0, 10)
        print(f"[Rank {rank}][P2P] Data {data.items()[0][0].data()}")

    await asyncio.sleep(2)

    ## receive from all
    if rank == 0:
        for i in range(1, workers+1):
            data = await zcomm.recv(i, 20)
            print(f"[Rank {rank}][P2P] Data {data}")
           
    else:
        await zcomm.send(0, b'321', 20)
        print(f"[Rank {rank}][P2P] Send to 0 20")


    await asyncio.sleep(2)
    return None


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", help=f"rank", type=int)
    parser.add_argument("-w", help="number of workers", type=int)
    parser.add_argument("-l", help="zenoh locator", default="tcp/127.0.0.1:7447", type=str)
    args = parser.parse_args()

    asyncio.run(asyncio_main(args.r, args.w, args.l))