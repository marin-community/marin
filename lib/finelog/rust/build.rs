// Compile the finelog protos at build time into buffa message views + connect
// service stubs, using anthropics/connect-rust's build.rs path
// (`connectrpc_build::Config`).
//
// connectrpc-build shells out to a `protoc` binary to parse the .proto files
// (the README's "no binary plugins" refers to protoc-gen-* codegen plugins, not
// to protoc itself). We vendor protoc via `protoc-bin-vendored` and hand its
// path to connectrpc-build through the `PROTOC` env var, so the build needs no
// system protoc and pins a version new enough for our `edition = "2023"` protos.
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
}
