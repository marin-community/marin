use clap::Parser;
use std::net::SocketAddr;

// Reference the library crate
extern crate fray_rusty;
use fray_rusty::coordinator::run_coordinator_server;

#[derive(Parser, Debug)]
#[command(name = "coordinator")]
#[command(about = "Fray Rusty Coordinator - Central coordination server for distributed task execution", long_about = None)]
struct Args {
    /// Address to bind the coordinator server to
    #[arg(short, long, default_value = "127.0.0.1:50051")]
    bind: SocketAddr,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args = Args::parse();

    log::info!("Starting Fray Rusty Coordinator");
    log::info!("Binding to: {}", args.bind);

    run_coordinator_server(args.bind).await?;

    Ok(())
}
