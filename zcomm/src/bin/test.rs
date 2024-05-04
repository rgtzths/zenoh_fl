//
// Copyright (c) 2024 Gabriele Baldoni
//
// Contributors:
//   Gabriele baldoni, <gabriele@zettascale.tech>
//

use clap::Parser;
use zcomm::{Error, ZComm};

#[derive(Parser, Clone, PartialEq, Eq, Hash, Debug)]
struct Args {
    rank: i8,
    workers: i8,
    locator: String,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();

    let zcomm = ZComm::new(args.rank, args.workers, args.locator).await?;
    let _ = zcomm.start().await;

    println!("Waiting for all nodes");

    zcomm.wait().await?;

    println!("All nodes discovered");

    Ok(())
}
