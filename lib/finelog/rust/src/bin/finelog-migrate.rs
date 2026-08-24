use std::path::PathBuf;

use clap::{Parser, Subcommand};
use finelog::migrations::telemetry_v1::{prepare_store, verify_store, PrepareConfig};

const DEFAULT_BATCH_ROWS: usize = 65_536;

#[derive(Parser)]
#[command(name = "finelog-migrate")]
#[command(about = "Prepare and verify offline Finelog store migrations")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build a replacement store with legacy telemetry split into physical shards.
    PrepareTelemetryV1 {
        /// Frozen source store. The command never modifies it.
        #[arg(long)]
        source_dir: PathBuf,
        /// Replacement store to build. It must differ from the source.
        #[arg(long)]
        output_dir: PathBuf,
        /// Path the replacement will occupy after the atomic swap.
        #[arg(long)]
        final_log_dir: PathBuf,
        /// Rows decoded from a source Parquet segment at a time.
        #[arg(long, default_value_t = DEFAULT_BATCH_ROWS)]
        batch_rows: usize,
    },
    /// Recheck a prepared replacement store without changing it.
    VerifyTelemetryV1 {
        #[arg(long)]
        source_dir: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value_t = DEFAULT_BATCH_ROWS)]
        batch_rows: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let manifest = match args.command {
        Command::PrepareTelemetryV1 {
            source_dir,
            output_dir,
            final_log_dir,
            batch_rows,
        } => prepare_store(&PrepareConfig {
            source_dir,
            output_dir,
            final_log_dir,
            batch_rows,
        })?,
        Command::VerifyTelemetryV1 {
            source_dir,
            output_dir,
            batch_rows,
        } => verify_store(&source_dir, &output_dir, batch_rows)?,
    };
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}
