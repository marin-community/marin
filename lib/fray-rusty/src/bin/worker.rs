use clap::Parser;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::time;

// Reference the library crate
extern crate fray_rusty;
use fray_rusty::worker::CoordinatorClient;
use fray_rusty::WorkerId;

#[derive(Parser, Debug)]
#[command(name = "worker")]
#[command(about = "Fray Rusty Worker - Execute tasks and host actors", long_about = None)]
struct Args {
    /// Coordinator address to connect to
    #[arg(short, long)]
    coordinator: SocketAddr,

    /// Worker capacity (number of concurrent tasks)
    #[arg(long, default_value = "4")]
    capacity: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();
    let worker_id = WorkerId::new();

    log::info!("Starting Fray Rusty Worker");
    log::info!("Worker ID: {}", worker_id);
    log::info!("Connecting to coordinator: {}", args.coordinator);
    log::info!("Capacity: {}", args.capacity);

    // Connect to coordinator
    let client = CoordinatorClient::connect(args.coordinator).await?;

    // Register worker
    client.register_worker(worker_id).await?;
    log::info!("Registered with coordinator");

    // Start heartbeat loop
    let heartbeat_client = client.clone();
    let heartbeat_worker_id = worker_id;
    tokio::spawn(async move {
        let mut interval = time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            if let Err(e) = heartbeat_client.heartbeat(heartbeat_worker_id).await {
                log::error!("Heartbeat failed: {}", e);
            }
        }
    });

    // Keep worker alive
    log::info!("Worker running. Press Ctrl+C to stop.");
    tokio::signal::ctrl_c().await?;
    log::info!("Shutting down worker");

    Ok(())
}
