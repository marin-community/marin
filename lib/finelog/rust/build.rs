// Compile the finelog protos at build time into buffa message views + connect
// service stubs, using anthropics/connect-rust's build.rs path
// (`connectrpc_build::Config`), and stamp the source revision into the binary.
//
// connectrpc-build shells out to a `protoc` binary to parse the .proto files
// (the README's "no binary plugins" refers to protoc-gen-* codegen plugins, not
// to protoc itself). We vendor protoc via `protoc-bin-vendored` and hand its
// path to connectrpc-build through the `PROTOC` env var, so the build needs no
// system protoc and pins a version new enough for our `edition = "2023"` protos.

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc binary");
    println!(
        "cargo:warning=finelog: using vendored protoc at {}",
        protoc.display()
    );
    // SAFETY: build scripts are single-threaded; this just sets the env var
    // connectrpc-build reads to locate protoc.
    std::env::set_var("PROTOC", &protoc);

    connectrpc_build::Config::new()
        .files(&["proto/logging.proto", "proto/finelog_stats.proto"])
        .includes(&["proto/"])
        .include_file("_connectrpc.rs")
        .compile()
        .unwrap();

    emit_build_stamp();
}

/// Bake the source revision and build environment into the binary as
/// `FINELOG_BUILD_*` env vars, which `server::build_info` reads back.
///
/// The revision comes from `FINELOG_SOURCE_*` when set, else from git. The
/// override exists because the deploy image copies only `lib/finelog/rust` into
/// its build stage — there is no checkout to interrogate there, so the builder
/// passes the revision it built from as a `--build-arg`.
///
/// Every value degrades to an empty string rather than failing the build: a
/// wheel built from an sdist has neither, and a build stamp is diagnostic, not
/// load-bearing.
fn emit_build_stamp() {
    let commit = source_var("FINELOG_SOURCE_COMMIT").unwrap_or_else(|| git(&["rev-parse", "HEAD"]));
    // The tree hash names the source CONTENT, so two commits that differ only in
    // message or parent — a rebase, a cherry-pick — are visibly the same build.
    let tree =
        source_var("FINELOG_SOURCE_TREE").unwrap_or_else(|| git(&["rev-parse", "HEAD^{tree}"]));
    let dirty = match source_var("FINELOG_SOURCE_DIRTY") {
        Some(flag) => flag == "true",
        None => !git(&["status", "--porcelain"]).is_empty(),
    };
    let built_at_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    println!("cargo:rustc-env=FINELOG_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=FINELOG_BUILD_TREE={tree}");
    println!("cargo:rustc-env=FINELOG_BUILD_DIRTY={dirty}");
    println!("cargo:rustc-env=FINELOG_BUILD_AT_UNIX={built_at_unix}");
    println!("cargo:rustc-env=FINELOG_BUILD_RUSTC={}", rustc_version());

    // Emitting any rerun-if-changed narrows cargo from "rerun when any package
    // file changes" to exactly this list, so the package inputs have to be named
    // alongside HEAD — otherwise editing a source file would recompile the crate
    // while keeping the previous stamp, and `dirty` would lie.
    for input in ["src", "proto", "Cargo.toml"] {
        println!("cargo:rerun-if-changed={input}");
    }
    for var in [
        "FINELOG_SOURCE_COMMIT",
        "FINELOG_SOURCE_TREE",
        "FINELOG_SOURCE_DIRTY",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    // `--git-path` resolves through a worktree's `.git` file, where the literal
    // path `.git/HEAD` does not exist.
    let head = git(&["rev-parse", "--git-path", "HEAD"]);
    if !head.is_empty() {
        println!("cargo:rerun-if-changed={head}");
    }
}

/// A `FINELOG_SOURCE_*` override, or `None` when unset or empty. An empty value
/// is treated as absent because an unset `--build-arg` reaches the build as one.
fn source_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// Trimmed stdout of `git <args>`, or an empty string if git is absent, the
/// directory is not a repository, or the command fails.
fn git(args: &[&str]) -> String {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_default()
}

/// First line of `rustc -vV`, e.g. `rustc 1.90.0 (1159e78c4 2025-09-14)`.
fn rustc_version() -> String {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    Command::new(rustc)
        .arg("-vV")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .map(str::to_string)
        })
        .unwrap_or_default()
}
