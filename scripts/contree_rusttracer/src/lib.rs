use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const SKIP_MARKERS: &[&str] = &[
    "site-packages",
    "lib/python",
    "/usr/lib",
    "importlib",
    "<frozen",
    "/pytracer/",
    "/.local/",
    "/usr/local/lib/",
    "_pytest/",
    "pluggy/",
    "conftest.py",
    "_distutils_hack",
    "pkg_resources",
];

#[pyclass]
struct Tracer {
    output: BufWriter<File>,
    repo_root: PathBuf,
    active_test: Option<String>,
    active_file: String,
    active_function: String,
    trace: String,
    event_count: u64,
    final_depth_cap: i64,
}

#[pymethods]
impl Tracer {
    #[new]
    fn new(output_path: &str, repo_root: &str) -> PyResult<Self> {
        let output = File::create(output_path)?;
        Ok(Self {
            output: BufWriter::new(output),
            repo_root: PathBuf::from(repo_root),
            active_test: None,
            active_file: String::new(),
            active_function: String::new(),
            trace: String::new(),
            event_count: 0,
            final_depth_cap: -1,
        })
    }

    fn trace(
        &mut self,
        py: Python<'_>,
        frame: &Bound<'_, PyAny>,
        event: &str,
        _arg: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if event != "line" {
            return Ok(());
        }

        let filename = frame.getattr("f_code")?.getattr("co_filename")?.extract::<String>()?;
        if self.should_skip(&filename) {
            return Ok(());
        }

        let test_id = match current_pytest_nodeid(py)? {
            Some(test_id) => test_id,
            None => return Ok(()),
        };
        if self.active_test.as_deref() != Some(test_id.as_str()) {
            self.finish_active()?;
            self.active_test = Some(test_id);
            self.active_file = relative_path(&self.repo_root, &filename);
            self.active_function = frame
                .getattr("f_code")?
                .getattr("co_name")?
                .extract::<String>()?;
            self.trace.clear();
            self.event_count = 0;
        }

        let lineno = frame.getattr("f_lineno")?.extract::<usize>()?;
        self.event_count += 1;
        self.trace.push_str(&format!(
            "{}:{} in {}\n",
            self.active_file, lineno, self.active_function
        ));
        if let Some(source) = source_line(&filename, lineno) {
            self.trace.push_str(source.trim_end());
            self.trace.push('\n');
        }
        Ok(())
    }

    fn finish(&mut self) -> PyResult<()> {
        self.finish_active()?;
        self.output.flush()?;
        Ok(())
    }
}

impl Tracer {
    fn should_skip(&self, filename: &str) -> bool {
        if SKIP_MARKERS.iter().any(|marker| filename.contains(marker)) {
            return true;
        }
        if filename.starts_with('<') {
            return true;
        }
        let path = Path::new(filename);
        !path.starts_with(&self.repo_root)
    }

    fn finish_active(&mut self) -> PyResult<()> {
        let Some(test_id) = self.active_test.take() else {
            return Ok(());
        };
        let trace = format!("# --- test source ---\n# --- execution trace ---\n{}", self.trace);
        let escaped_test_id = json_escape(&test_id);
        let escaped_file = json_escape(&self.active_file);
        let escaped_function = json_escape(&self.active_function);
        let escaped_trace = json_escape(&trace);
        let row = format!(
            "{{\"test_id\":\"{}\",\"file\":\"{}\",",
            escaped_test_id, escaped_file
        ) + &format!(
            "\"function\":\"{}\",\"trace\":\"{}\",",
            escaped_function, escaped_trace
        ) + &format!(
            "\"event_count\":{},\"final_depth_cap\":{}}}\n",
            self.event_count, self.final_depth_cap
        );
        self.output.write_all(row.as_bytes())?;
        Ok(())
    }
}

fn current_pytest_nodeid(py: Python<'_>) -> PyResult<Option<String>> {
    let os = py.import("os")?;
    let environ = os.getattr("environ")?.downcast_into::<PyDict>()?;
    let Some(current) = environ.get_item("PYTEST_CURRENT_TEST")? else {
        return Ok(None);
    };
    let current = current.extract::<String>()?;
    let nodeid = current.rsplit_once(' ').map(|(nodeid, _)| nodeid).unwrap_or(&current);
    Ok(Some(nodeid.to_string()))
}

fn relative_path(repo_root: &Path, filename: &str) -> String {
    let path = Path::new(filename);
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn source_line(filename: &str, lineno: usize) -> Option<String> {
    let contents = std::fs::read_to_string(filename).ok()?;
    contents.lines().nth(lineno.checked_sub(1)?).map(str::to_string)
}

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ch if ch < ' ' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

#[pymodule]
fn _contree_rusttracer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tracer>()?;
    Ok(())
}
