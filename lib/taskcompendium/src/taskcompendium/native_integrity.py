# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Language-specific runner, configuration and dependency boundaries."""

from dataclasses import dataclass

from taskcompendium.integrity import SharedTestFamily


@dataclass(frozen=True)
class NativeFamily:
    tools: tuple[str, ...]
    configs: tuple[str, ...]
    dependencies: tuple[str, ...]
    hooks: tuple[str, ...]
    environment: tuple[str, ...] = ()
    runner_paths: tuple[str, ...] = ()


FAMILIES = {
    SharedTestFamily.GO: NativeFamily(
        ("go",),
        ("**/go.mod", "**/go.sum", "**/go.work*", "**/*_test.go"),
        ("~/go/pkg/mod",),
        ("**/go.env",),
        ("GOPATH", "GOMODCACHE", "GOPROXY", "GONOSUMDB", "GOPRIVATE"),
    ),
    SharedTestFamily.RUST: NativeFamily(
        ("cargo", "rustc", "rustdoc"),
        ("**/Cargo.toml", "**/Cargo.lock", "**/.cargo/**", "**/rust-toolchain*", "**/build.rs"),
        ("~/.cargo/registry", "~/.cargo/git"),
        ("**/.cargo/config*", "**/build.rs"),
        ("CARGO_HOME", "RUSTUP_HOME"),
    ),
    SharedTestFamily.JVM: NativeFamily(
        ("java", "javac", "mvn", "gradle", "ant", "sbt"),
        (
            "**/pom.xml",
            "**/build.gradle*",
            "**/settings.gradle*",
            "**/gradle.properties",
            "**/gradlew*",
            "**/gradle/**",
            "**/.mvn/**",
            "**/mvnw*",
            "**/build.xml",
            "**/build.sbt",
            "**/project/**",
        ),
        ("~/.m2", "~/.gradle", "~/.ivy2/cache"),
        ("**/init.gradle*", "**/init.d/**", "**/settings.xml", "**/jvm.config", "**/maven.config"),
        ("JAVA_HOME", "MAVEN_HOME", "GRADLE_HOME"),
        runner_paths=(
            "~/.gradle/wrapper/dists",
            "~/.m2/repository/org/apache/maven",
            "~/.m2/repository/org/junit",
            "~/.m2/repository/junit",
            "~/.m2/repository/org/testng",
            "~/.gradle/caches/modules-2/files-2.1/org.junit*",
            "~/.gradle/caches/modules-2/files-2.1/junit",
            "~/.gradle/caches/modules-2/files-2.1/org.testng",
            "/usr/share/java/junit*",
        ),
    ),
    SharedTestFamily.JULIA: NativeFamily(
        ("julia",),
        ("**/Project.toml", "**/Manifest.toml"),
        ("~/.julia",),
        ("**/startup.jl",),
        ("JULIA_DEPOT_PATH",),
    ),
    SharedTestFamily.SWIFT: NativeFamily(
        ("swift", "swiftc"),
        ("**/Package.swift", "**/Package.resolved"),
        ("~/.cache/org.swift.swiftpm", "{workspace}/.build/checkouts"),
        ("**/.swiftpm/configuration/**",),
    ),
    SharedTestFamily.CPP: NativeFamily(
        ("cc", "c++", "gcc", "g++", "clang", "clang++", "cmake", "ctest", "make", "ninja"),
        ("**/CMakeLists.txt", "**/*.cmake", "**/CMakePresets.json", "**/Makefile*", "**/meson.build"),
        ("/usr/local/include", "{workspace}/third_party"),
        ("**/CTestCustom.cmake", "**/.cmake/**"),
    ),
    SharedTestFamily.DART: NativeFamily(
        ("dart", "flutter"),
        ("**/pubspec.yaml", "**/pubspec.lock", "**/dart_test.yaml"),
        ("~/.pub-cache", "~/.dart-tool"),
        ("**/.dart_tool/package_config.json",),
        ("PUB_CACHE", "FLUTTER_ROOT"),
        runner_paths=(
            "~/.pub-cache/hosted/*/test-*",
            "~/.pub-cache/hosted/*/test_core-*",
            "~/.pub-cache/hosted/*/test_api-*",
            "~/.pub-cache/hosted/*/matcher-*",
        ),
    ),
    SharedTestFamily.PHP: NativeFamily(
        ("php", "phpunit", "composer"),
        ("**/phpunit.xml*", "**/composer.json", "**/composer.lock"),
        ("{workspace}/vendor",),
        ("**/.user.ini", "**/php.ini"),
        runner_paths=(
            "{workspace}/vendor/phpunit",
            "{workspace}/vendor/sebastian",
            "/usr/share/php/PHPUnit",
            "/usr/share/php/SebastianBergmann",
        ),
    ),
    SharedTestFamily.ELIXIR: NativeFamily(
        ("elixir", "erl", "mix"),
        ("**/mix.exs", "**/mix.lock", "**/config/**"),
        ("{workspace}/deps", "~/.mix/archives"),
        ("**/.iex.exs",),
        ("ERL_ROOTDIR",),
    ),
    SharedTestFamily.DOTNET: NativeFamily(
        ("dotnet",),
        (
            "**/*.csproj",
            "**/*.fsproj",
            "**/*.sln",
            "**/Directory.Build.*",
            "**/global.json",
            "**/NuGet.Config",
            "**/*.runsettings",
        ),
        ("~/.nuget/packages", "~/.dotnet"),
        ("**/Directory.Build.*",),
        ("DOTNET_ROOT",),
        runner_paths=(
            "~/.nuget/packages/microsoft.net.test.sdk",
            "~/.nuget/packages/microsoft.testplatform.*",
            "~/.nuget/packages/xunit*",
            "~/.nuget/packages/nunit*",
            "~/.nuget/packages/mstest.*",
        ),
    ),
    SharedTestFamily.LUA: NativeFamily(
        ("lua", "luajit", "nvim", "busted", "luarocks"),
        ("**/.busted", "**/*.rockspec", "**/Makefile*"),
        ("~/.luarocks", "{workspace}/lua_modules"),
        ("**/init.lua", "**/.luarc.json", "**/.luarocks/config*"),
        runner_paths=("~/.luarocks/share/lua/*/busted", "~/.luarocks/share/lua/*/luassert"),
    ),
    SharedTestFamily.OCAML: NativeFamily(
        ("ocaml", "ocamlc", "ocamlopt", "dune", "opam"),
        ("**/dune", "**/dune-project", "**/*.opam"),
        ("~/.opam/download-cache", "~/.opam/default/lib"),
        ("**/.ocamlinit",),
        ("OPAMROOT", "OPAM_SWITCH_PREFIX"),
    ),
}
LANGUAGES = {
    "go": SharedTestFamily.GO,
    "rust": SharedTestFamily.RUST,
    "java": SharedTestFamily.JVM,
    "kotlin": SharedTestFamily.JVM,
    "scala": SharedTestFamily.JVM,
    "julia": SharedTestFamily.JULIA,
    "swift": SharedTestFamily.SWIFT,
    "c": SharedTestFamily.CPP,
    "cpp": SharedTestFamily.CPP,
    "c++": SharedTestFamily.CPP,
    "dart": SharedTestFamily.DART,
    "php": SharedTestFamily.PHP,
    "elixir": SharedTestFamily.ELIXIR,
    "csharp": SharedTestFamily.DOTNET,
    "c#": SharedTestFamily.DOTNET,
    "lua": SharedTestFamily.LUA,
    "ocaml": SharedTestFamily.OCAML,
}
TEST_PATTERNS = (
    "**/tests/**",
    "**/test/**",
    "**/Tests/**",
    "**/testing/**",
    "**/testdata/**",
    "**/test-data/**",
    "**/spec/**",
    "**/specs/**",
    "**/test_*",
    "**/*_test.*",
    "**/*.test.*",
    "**/*.spec.*",
)
