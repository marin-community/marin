use std::path::PathBuf;

use clap::{Parser, Subcommand};
use finelog::migrations::telemetry_v1::{
    prepare_in_place, publish_in_place, retire_in_place, verify_in_place, InPlaceConfig,
};

const DEFAULT_BATCH_ROWS: usize = 65_536;

#[derive(Parser)]
#[command(name = "finelog-migrate")]
#[command(about = "Stage and cut over Finelog store migrations")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Snapshot root telemetry and stage physical shards inside the same store.
    PrepareTelemetryV1 {
        /// Finelog store containing the catalog and namespace directories.
        #[arg(long)]
        store_dir: PathBuf,
        /// Rows decoded from a source Parquet segment at a time.
        #[arg(long, default_value_t = DEFAULT_BATCH_ROWS)]
        batch_rows: usize,
    },
    /// Publish staged shards in a new catalog. Finelog must be stopped.
    PublishTelemetryV1 {
        #[arg(long)]
        store_dir: PathBuf,
        #[arg(long, default_value_t = DEFAULT_BATCH_ROWS)]
        batch_rows: usize,
    },
    /// Remove the root namespace after queries use semantic names. Finelog must be stopped.
    RetireTelemetryV1 {
        #[arg(long)]
        store_dir: PathBuf,
    },
    /// Recheck the current migration phase without changing it.
    VerifyTelemetryV1 {
        #[arg(long)]
        store_dir: PathBuf,
        #[arg(long, default_value_t = DEFAULT_BATCH_ROWS)]
        batch_rows: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let manifest = match args.command {
        Command::PrepareTelemetryV1 {
            store_dir,
            batch_rows,
        } => prepare_in_place(&InPlaceConfig {
            store_dir,
            batch_rows,
        })?,
        Command::PublishTelemetryV1 {
            store_dir,
            batch_rows,
        } => publish_in_place(&InPlaceConfig {
            store_dir,
            batch_rows,
        })?,
        Command::RetireTelemetryV1 { store_dir } => retire_in_place(&InPlaceConfig {
            store_dir,
            batch_rows: DEFAULT_BATCH_ROWS,
        })?,
        Command::VerifyTelemetryV1 {
            store_dir,
            batch_rows,
        } => verify_in_place(&InPlaceConfig {
            store_dir,
            batch_rows,
        })?,
    };
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
