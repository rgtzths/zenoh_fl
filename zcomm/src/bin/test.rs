//
// Copyright (c) 2024 Gabriele Baldoni
//
// Contributors:
//   Gabriele baldoni, <gabriele@zettascale.tech>
//

use anyhow::Error;
use clap::Parser;
use std::{sync::Arc, time::Duration};
use zcomm::{ZComm, ALL_SRC, ANY_SRC};

#[derive(Parser, Clone, PartialEq, Eq, Hash, Debug)]
struct Args {
    rank: i8,
    workers: i8,
    locator: String,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();

    let zcomm = ZComm::new(args.rank, args.workers, args.locator)
        .await
        .unwrap();
    let _ = zcomm.start();

    println!("Waiting for all nodes");

    zcomm.wait().await.unwrap();

    println!("All nodes discovered");

    // send to all nodes (bcast)

    let data = zcomm.bcast(0, Arc::new(vec![1, 2, 3]), 100).await;

    println!("[Rank {}][BCAST] Data: {:?}", args.rank, data);

    tokio::time::sleep(Duration::from_secs(1)).await;

    // send to all nodes (p2p)
    if args.rank == 0 {
        for i in 1..args.workers + 1 {
            zcomm.send(i, Arc::new(vec![3, 2, 1]), 10).await.unwrap();
            println!("[Rank {}][P2P] Send to {} 10", args.rank, i);
        }
    } else {
        let data = zcomm.recv(0, 10).await.unwrap();
        println!("[Rank {}][P2P] Data: {:?}", args.rank, data);
    }

    tokio::time::sleep(Duration::from_secs(1)).await;

    // receive from all
    if args.rank == 0 {
        for i in 1..args.workers + 1 {
            let data = zcomm.recv(i, 20).await.unwrap();
            println!("[Rank {}][P2P] Data: {:?}", args.rank, data);
        }
    } else {
        zcomm.send(0, Arc::new(vec![9, 8, 7]), 20).await.unwrap();
        println!("[Rank {}][P2P] Send to {} 20", args.rank, 0);
    }

    // receive from all
    if args.rank == 0 {
        let data = zcomm.recv(ALL_SRC, 15).await.unwrap();
        println!("[Rank {}][P2P] Data: {:?}", args.rank, data);
    } else {
        zcomm.send(0, Arc::new(vec![5, 4, 3]), 15).await.unwrap();
        println!("[Rank {}][P2P] Send to {} 15", args.rank, 0);
    }

    tokio::time::sleep(Duration::from_secs(1)).await;

    // receive from any
    if args.rank == 0 {
        let data = zcomm.recv(ANY_SRC, 30).await.unwrap();
        println!("[Rank {}][P2P] Data: {:?}", args.rank, data);
    } else {
        zcomm.send(0, Arc::new(vec![3, 4, 5]), 30).await.unwrap();
        println!("[Rank {}][P2P] Send to {} 30", args.rank, 0);
    }

    tokio::time::sleep(Duration::from_secs(3)).await;

    Ok(())
}
