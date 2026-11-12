# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small real-language projects for the manual Linux integrity regression matrix."""

from dataclasses import dataclass

from taskcompendium.integrity import SharedTestFamily


@dataclass(frozen=True)
class NativeProject:
    family: SharedTestFamily
    tools: tuple[str, ...]
    command: str
    config: str
    files: dict[str, str]
    dependency_files: dict[str, str]
    mutation_file: str
    mutation: str


PROJECTS = {
    "go": NativeProject(
        SharedTestFamily.GO,
        ("go",),
        "go test -count=1 -v ./...",
        "go.mod",
        {
            "go.mod": (
                "module example.test/candidate\ngo 1.19\nrequire example.test/dependency v0.0.0\n"
                "replace example.test/dependency => {dependency}\n"
            ),
            "value.go": "package candidate\nconst Value = 0\n",
            "tests/value_test.go": (
                'package tests\nimport ("testing"; d "example.test/dependency")\n'
                'func TestValue(t *testing.T) { if d.Value != 1 { t.Fatal("broken") } }\n'
            ),
        },
        {"go.mod": "module example.test/dependency\ngo 1.19\n", "value.go": "package dependency\nconst Value = 0\n"},
        "value.go",
        "package dependency\nconst Value = 1\n",
    ),
    "rust": NativeProject(
        SharedTestFamily.RUST,
        ("cargo", "rustc"),
        "cargo test --offline",
        "Cargo.toml",
        {
            "Cargo.toml": (
                '[package]\nname="candidate"\nversion="0.1.0"\nedition="2021"\n'
                '[dependencies]\ndependency={path="{dependency}"}\n'
            ),
            "src/lib.rs": "pub fn value() -> u32 { dependency::value() }\n",
            "tests/value.rs": "#[test]\nfn test_value() { assert_eq!(candidate::value(), 1); }\n",
        },
        {
            "Cargo.toml": '[package]\nname="dependency"\nversion="0.1.0"\nedition="2021"\n',
            "src/lib.rs": "pub fn value() -> u32 { 0 }\n",
        },
        "src/lib.rs",
        "pub fn value() -> u32 { 1 }\n",
    ),
    "gradle": NativeProject(
        SharedTestFamily.JVM,
        ("java", "javac", "gradle"),
        "gradle --offline --no-daemon test",
        "build.gradle",
        {
            "build.gradle": (
                "apply plugin: 'java'\nsourceSets.main.java.srcDirs += ['{dependency}']\n"
                "dependencies { testImplementation files('/usr/share/java/junit4.jar') }\n"
            ),
            "src/test/java/ValueTest.java": (
                "import org.junit.Test; import static org.junit.Assert.*;\n"
                "public class ValueTest { @Test public void value() { assertEquals(1, Value.get()); } }\n"
            ),
        },
        {"Value.java": "public class Value { public static int get() { return 0; } }\n"},
        "Value.java",
        "public class Value { public static int get() { return 1; } }\n",
    ),
    "maven": NativeProject(
        SharedTestFamily.JVM,
        ("java", "javac", "mvn"),
        "mvn -q test",
        "pom.xml",
        {
            "pom.xml": (
                '<project xmlns="http://maven.apache.org/POM/4.0.0"><modelVersion>4.0.0</modelVersion>'
                "<groupId>test</groupId><artifactId>candidate</artifactId><version>1</version>"
                "<properties><maven.compiler.source>8</maven.compiler.source>"
                "<maven.compiler.target>8</maven.compiler.target></properties>"
                "<build><sourceDirectory>{dependency}</sourceDirectory></build><dependencies>"
                "<dependency><groupId>junit</groupId><artifactId>junit</artifactId><version>4.13.2</version>"
                "<scope>test</scope></dependency></dependencies></project>"
            ),
            "src/test/java/ValueTest.java": (
                "import org.junit.Test; import static org.junit.Assert.*;\n"
                "public class ValueTest { @Test public void value() { assertEquals(1, Value.get()); } }\n"
            ),
        },
        {"Value.java": "public class Value { public static int get() { return 0; } }\n"},
        "Value.java",
        "public class Value { public static int get() { return 1; } }\n",
    ),
    "cpp": NativeProject(
        SharedTestFamily.CPP,
        ("g++", "cmake", "ctest", "make"),
        "cmake -S . -B build && cmake --build build && ctest --test-dir build --output-on-failure",
        "CMakeLists.txt",
        {
            "CMakeLists.txt": (
                "cmake_minimum_required(VERSION 3.16)\nproject(candidate)\nenable_testing()\n"
                'add_executable(value tests/value.cpp)\ntarget_include_directories(value PRIVATE "{dependency}")\n'
                "add_test(NAME value COMMAND value)\n"
            ),
            "tests/value.cpp": '#include "value.h"\nint main() { return value() == 1 ? 0 : 1; }\n',
        },
        {"value.h": "inline int value() { return 0; }\n"},
        "value.h",
        "inline int value() { return 1; }\n",
    ),
    "php": NativeProject(
        SharedTestFamily.PHP,
        ("php", "phpunit"),
        "phpunit tests/ValueTest.php",
        "phpunit.xml",
        {
            "phpunit.xml": "<phpunit/>\n",
            "tests/ValueTest.php": (
                '<?php\nrequire "{dependency}/value.php";\n'
                "final class ValueTest extends PHPUnit\\Framework\\TestCase {\n"
                "public function testValue(): void { $this->assertSame(1, value()); } }\n"
            ),
        },
        {"value.php": "<?php function value() { return 0; }\n"},
        "value.php",
        "<?php function value() { return 1; }\n",
    ),
    "elixir": NativeProject(
        SharedTestFamily.ELIXIR,
        ("elixir", "erl", "mix"),
        "mix test",
        "mix.exs",
        {
            "mix.exs": (
                "defmodule Candidate.MixProject do\nuse Mix.Project\n"
                'def project, do: [app: :candidate, version: "0.1.0", elixirc_paths: ["{dependency}"]]\nend\n'
            ),
            "test/test_helper.exs": "ExUnit.start()\n",
            "test/value_test.exs": (
                'defmodule ValueTest do\nuse ExUnit.Case\ntest "value" do\n' "assert Value.get() == 1\nend\nend\n"
            ),
        },
        {"value.ex": "defmodule Value do\ndef get, do: 0\nend\n"},
        "value.ex",
        "defmodule Value do\ndef get, do: 1\nend\n",
    ),
    "lua": NativeProject(
        SharedTestFamily.LUA,
        ("lua", "busted"),
        "busted tests/value_spec.lua",
        ".busted",
        {
            ".busted": "return {}\n",
            "tests/value_spec.lua": (
                'package.path = "{dependency}/?.lua;" .. package.path\n'
                'describe("value", function() it("is one", function() assert.equals(1, require("value")) end) end)\n'
            ),
        },
        {"value.lua": "return 0\n"},
        "value.lua",
        "return 1\n",
    ),
    "ocaml": NativeProject(
        SharedTestFamily.OCAML,
        ("ocamlc", "dune"),
        "dune runtest",
        "dune-project",
        {
            "dune-project": "(lang dune 2.9)\n",
            "tests/dune": "(test (name value_test) (libraries dependency))\n",
            "tests/value_test.ml": "let () = assert (Dependency.value = 1)\n",
            "dependency": "",
        },
        {"dune": "(library (name dependency))\n", "dependency.ml": "let value = 0\n"},
        "dependency.ml",
        "let value = 1\n",
    ),
    "julia": NativeProject(
        SharedTestFamily.JULIA,
        ("julia",),
        "julia --startup-file=no tests/value.jl",
        "Project.toml",
        {
            "Project.toml": "[deps]\n",
            "tests/value.jl": 'using Test\ninclude("{dependency}/value.jl")\n@test value() == 1\n',
        },
        {"value.jl": "value() = 0\n"},
        "value.jl",
        "value() = 1\n",
    ),
    "swift": NativeProject(
        SharedTestFamily.SWIFT,
        ("swift", "swiftc"),
        "swift test",
        "Package.swift",
        {
            "Package.swift": (
                "// swift-tools-version: 5.9\nimport PackageDescription\n"
                'let package = Package(name: "Candidate", dependencies: [.package(path: "{dependency}")], targets: ['
                '.testTarget(name: "ValueTests", dependencies: ['
                '.product(name: "Dependency", package: "dependencies")])])\n'
            ),
            "Tests/ValueTests/ValueTests.swift": (
                "import XCTest\nimport Dependency\n"
                "final class ValueTests: XCTestCase { func testValue() { XCTAssertEqual(value(), 1) } }\n"
            ),
        },
        {
            "Package.swift": (
                "// swift-tools-version: 5.9\nimport PackageDescription\n"
                'let package = Package(name: "Dependency", '
                'products: [.library(name: "Dependency", targets: ["Dependency"])], '
                'targets: [.target(name: "Dependency")])\n'
            ),
            "Sources/Dependency/Value.swift": "public func value() -> Int { 0 }\n",
        },
        "Sources/Dependency/Value.swift",
        "public func value() -> Int { 1 }\n",
    ),
    "dart": NativeProject(
        SharedTestFamily.DART,
        ("dart",),
        "dart pub get && dart test",
        "pubspec.yaml",
        {
            "pubspec.yaml": (
                "name: candidate\nenvironment:\n  sdk: '>=3.0.0 <4.0.0'\n"
                "dependencies:\n  dependency:\n    path: {dependency}\ndev_dependencies:\n  test: ^1.25.0\n"
            ),
            "test/value_test.dart": (
                "import 'package:test/test.dart'; import 'package:dependency/value.dart';\n"
                "void main() { test('value', () { expect(value(), 1); }); }\n"
            ),
        },
        {
            "pubspec.yaml": "name: dependency\nenvironment:\n  sdk: '>=3.0.0 <4.0.0'\n",
            "lib/value.dart": "int value() => 0;\n",
        },
        "lib/value.dart",
        "int value() => 1;\n",
    ),
    "dotnet": NativeProject(
        SharedTestFamily.DOTNET,
        ("dotnet",),
        "dotnet test --nologo",
        "Candidate.csproj",
        {
            "Candidate.csproj": (
                '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net8.0</TargetFramework>'
                "<IsTestProject>true</IsTestProject></PropertyGroup><ItemGroup>"
                '<Compile Include="{dependency}/Value.cs"/>'
                '<PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.11.1"/>'
                '<PackageReference Include="xunit" Version="2.9.2"/>'
                '<PackageReference Include="xunit.runner.visualstudio" Version="2.8.2"/>'
                "</ItemGroup></Project>"
            ),
            "tests/ValueTest.cs": (
                "using Xunit; public class ValueTest { [Fact] public void TestValue() { "
                "Assert.Equal(1, Value.Get()); } }\n"
            ),
        },
        {"Value.cs": "public static class Value { public static int Get() => 0; }\n"},
        "Value.cs",
        "public static class Value { public static int Get() => 1; }\n",
    ),
}
