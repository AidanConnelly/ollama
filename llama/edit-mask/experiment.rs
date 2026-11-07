// Repeated, paired OpenCode/Ollama experiment for the experimental edit sampler.
//
// Compile:
//   rustc --edition=2021 llama/edit-mask/experiment.rs -o opencode-oldstring-mask-experiment
//
// Example:
//   go build -o /tmp/ollama-sampling-experiment .
//   ./opencode-oldstring-mask-experiment \
//     --repo path/to/sled \
//     --ollama /tmp/ollama-sampling-experiment \
//     --models /mnt/sda12/ollama \
//     --model 'hf.co/IFM/K2-Horizon-3.7B-GGUF:Q5_K_M' \
//     --task-file llama/edit-mask/sled-doc-comment-task.txt \
//     --n 10 \
//     --context-length 65536
//
// Each pair starts from the same clean Git commit in two separate worktrees.
// It starts a fresh Ollama server for each run, using the same binary and model
// store; the only treatment difference is OLLAMA_OPENCODE_EDIT_CONSTRAINTS.
// The task must exercise OpenCode's read-then-edit flow, and the selected
// model must use Ollama's native llama.cpp chat path for the treatment to apply.
// The CSV records clean process completion and the final self-report marker.
// Run score_edit.py separately to verify the requested edit and read sequence
// against the archived worktrees and transcripts.
//
// This is Linux/Unix oriented and uses only the Rust standard library.

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SUCCESS_MARKER: &str = "EXPERIMENT_RESULT: SUCCESS";
const FAILURE_MARKER: &str = "EXPERIMENT_RESULT: FAILURE";
const PROVIDER_ID: &str = "ollama-experiment";

unsafe extern "C" {
    fn kill(pid: i32, signal: i32) -> i32;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Condition {
    Control,
    Constrained,
}

impl Condition {
    fn name(self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Constrained => "constrained",
        }
    }

    fn flag_value(self) -> &'static str {
        match self {
            Self::Control => "0",
            Self::Constrained => "1",
        }
    }
}

struct Options {
    repo: PathBuf,
    ollama: PathBuf,
    opencode: PathBuf,
    models: PathBuf,
    model: String,
    task: String,
    pairs: usize,
    baseline: String,
    out: Option<PathBuf>,
    timeout: Duration,
    health_timeout: Duration,
    context_length: Option<usize>,
    seed: u64,
    alpha: f64,
    keep_worktrees: bool,
}

struct RunningOllama {
    child: Child,
    port: u16,
    log_path: PathBuf,
}

impl Drop for RunningOllama {
    fn drop(&mut self) {
        stop_process_group(&mut self.child);
    }
}

struct EventSummary {
    final_text_after_last_tool: String,
    error_events: usize,
    tool_errors: usize,
    open_steps: usize,
}

struct TrialResult {
    pair: usize,
    condition: Condition,
    order: &'static str,
    success: bool,
    self_report: &'static str,
    reason: String,
    elapsed_secs: f64,
    exit_code: String,
    timed_out: bool,
    tool_errors: usize,
    error_events: usize,
    open_steps: usize,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
    patch_path: PathBuf,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("experiment error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let Some(mut options) = parse_args()? else {
        return Ok(());
    };

    options.repo = canonicalize(&options.repo)?;
    options.models = canonicalize(&options.models)?;
    options.ollama = resolve_program(&options.ollama)?;
    options.opencode = resolve_program(&options.opencode)?;

    let repo_root = git_capture(&options.repo, &["rev-parse", "--show-toplevel"])?;
    let repo_root = canonicalize(Path::new(repo_root.trim()))?;
    options.repo = repo_root;

    let dirty = git_capture(
        &options.repo,
        &["status", "--porcelain", "--untracked-files=normal"],
    )?;
    if !dirty.trim().is_empty() {
        return Err(format!(
            "the baseline repository must be clean; commit or remove these changes first:\n{dirty}"
        ));
    }
    let base_commit = git_capture(
        &options.repo,
        &[
            "rev-parse",
            "--verify",
            &format!("{}^{{commit}}", options.baseline),
        ],
    )?
    .trim()
    .to_owned();

    let output = match options.out.take() {
        Some(path) => absolute_path(&path)?,
        None => {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| error.to_string())?
                .as_nanos();
            options.repo.parent().unwrap_or(&options.repo).join(format!(
                "opencode-oldstring-mask-experiment-{stamp}-{}",
                std::process::id()
            ))
        }
    };
    if output.starts_with(&options.repo) {
        return Err(format!(
            "output directory must be outside the Git repository: {}",
            output.display()
        ));
    }
    if output.exists()
        && fs::read_dir(&output)
            .map_err(display_error)?
            .next()
            .is_some()
    {
        return Err(format!(
            "output directory is not empty: {}",
            output.display()
        ));
    }
    fs::create_dir_all(&output).map_err(display_error)?;
    fs::create_dir_all(output.join("trials")).map_err(display_error)?;
    fs::create_dir_all(output.join("worktrees")).map_err(display_error)?;

    let ollama_version = command_version(&options.ollama, &["--version"])?;
    let opencode_version = command_version(&options.opencode, &["--version"])?;
    let context_length_record = options
        .context_length
        .map(|value| value.to_string())
        .or_else(|| env::var("OLLAMA_CONTEXT_LENGTH").ok())
        .unwrap_or_else(|| "Ollama default".to_owned());
    let prompt = scored_prompt(&options.task);
    write_text(&output.join("task.txt"), &options.task)?;
    write_text(&output.join("prompt.txt"), &prompt)?;
    write_text(
        &output.join("metadata.txt"),
        &format!(
            "baseline_commit={base_commit}\nmodel={}\npairs={}\nseed={}\ngeneration_seed={}\nalpha={}\nollama={}\nollama_version={}\nopencode={}\nopencode_version={}\nmodels={}\ncontext_length={}\nthinking_requested=false\nprovider_header_timeout=false\ntimeout_seconds={}\nimplementation=oldString-mask-v1\nworking_directory=identical absolute path for both conditions, archived after each run\nspeculative=false\nbackend_sampling=false\nscore=self-report retained in CSV; actual requested edit scored separately in actual-edit-score.json\n",
            options.model,
            options.pairs,
            options.seed,
            options.seed,
            options.alpha,
            options.ollama.display(),
            ollama_version.trim().replace('\n', " "),
            options.opencode.display(),
            opencode_version.trim().replace('\n', " "),
            options.models.display(),
            context_length_record,
            options.timeout.as_secs(),
        ),
    )?;

    let csv_path = output.join("results.csv");
    write_text(
        &csv_path,
        "pair,condition,order,baseline_commit,model,elapsed_secs,exit_code,timed_out,self_report,success,reason,tool_errors,error_events,open_steps,stdout,stderr,patch\n",
    )?;

    println!("Output: {}", output.display());
    println!("Baseline: {base_commit}");
    println!("Model: {}", options.model);
    println!(
        "Running {} paired comparisons ({} total runs)",
        options.pairs,
        options.pairs * 2
    );

    let mut order_seed = options.seed;
    let treatment_first_on_even = next_random(&mut order_seed) & 1 == 0;
    let mut results = Vec::with_capacity(options.pairs * 2);
    for pair_index in 0..options.pairs {
        let treatment_first = (pair_index % 2 == 0) == treatment_first_on_even;
        let order = if treatment_first {
            [Condition::Constrained, Condition::Control]
        } else {
            [Condition::Control, Condition::Constrained]
        };
        for condition in order {
            let pair_number = pair_index + 1;
            let order_name = if treatment_first {
                "constrained-first"
            } else {
                "control-first"
            };
            println!("Pair {pair_number}/{}: {}", options.pairs, condition.name());
            let result = run_trial(
                &options,
                &base_commit,
                &prompt,
                &output,
                pair_number,
                condition,
                order_name,
            )?;
            append_csv(&csv_path, &result, &base_commit, &options.model)?;
            println!(
                "  {} after {:.1}s ({})",
                if result.success { "SUCCESS" } else { "FAILURE" },
                result.elapsed_secs,
                result.reason
            );
            results.push(result);
        }
    }

    let summary = summarize(&results, options.pairs, options.alpha);
    write_text(&output.join("summary.txt"), &summary)?;
    println!("{summary}");
    Ok(())
}

fn parse_args() -> Result<Option<Options>, String> {
    let mut repo = None;
    let mut ollama = None;
    let mut opencode = PathBuf::from("opencode");
    let mut models = None;
    let mut model = None;
    let mut task_file = None;
    let mut task = None;
    let mut pairs = 10usize;
    let mut baseline = "HEAD".to_owned();
    let mut out = None;
    let mut timeout_secs = 900u64;
    let mut health_timeout_secs = 180u64;
    let mut context_length = None;
    let mut seed = 20261106u64;
    let mut alpha = 0.05f64;
    let mut keep_worktrees = false;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut index = 0usize;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--help" || arg == "-h" {
            print!("{}", usage());
            return Ok(None);
        }
        index += 1;
        if arg == "--keep-worktrees" {
            keep_worktrees = true;
            continue;
        }
        let value = args
            .get(index)
            .ok_or_else(|| format!("missing value for {arg}\n\n{}", usage()))?;
        index += 1;
        match arg.as_str() {
            "--repo" => repo = Some(PathBuf::from(value)),
            "--ollama" => ollama = Some(PathBuf::from(value)),
            "--opencode" => opencode = PathBuf::from(value),
            "--models" => models = Some(PathBuf::from(value)),
            "--model" => model = Some(value.clone()),
            "--task-file" => task_file = Some(PathBuf::from(value)),
            "--task" => task = Some(value.clone()),
            "--n" | "--pairs" => {
                pairs = value
                    .parse()
                    .map_err(|_| format!("invalid positive integer for {arg}: {value}"))?
            }
            "--baseline" => baseline = value.clone(),
            "--out" => out = Some(PathBuf::from(value)),
            "--timeout-secs" => {
                timeout_secs = value
                    .parse()
                    .map_err(|_| format!("invalid number for {arg}: {value}"))?
            }
            "--health-timeout-secs" => {
                health_timeout_secs = value
                    .parse()
                    .map_err(|_| format!("invalid number for {arg}: {value}"))?
            }
            "--context-length" => {
                context_length = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid positive integer for {arg}: {value}"))?,
                )
            }
            "--seed" => {
                seed = value
                    .parse()
                    .map_err(|_| format!("invalid integer for {arg}: {value}"))?
            }
            "--alpha" => {
                alpha = value
                    .parse()
                    .map_err(|_| format!("invalid probability for {arg}: {value}"))?
            }
            _ => return Err(format!("unknown option: {arg}\n\n{}", usage())),
        }
    }

    if pairs == 0 || timeout_secs == 0 || health_timeout_secs == 0 {
        return Err("--n, --timeout-secs, and --health-timeout-secs must be positive".into());
    }
    if !alpha.is_finite() || !(0.0..1.0).contains(&alpha) {
        return Err("--alpha must be greater than 0 and less than 1".into());
    }
    if context_length == Some(0) {
        return Err("--context-length must be positive".into());
    }
    if task.is_some() && task_file.is_some() {
        return Err("choose either --task or --task-file".into());
    }
    let task = match (task, task_file) {
        (Some(value), None) => value,
        (None, Some(path)) => fs::read_to_string(&path)
            .map_err(|error| format!("cannot read task file {}: {error}", path.display()))?,
        (None, None) => return Err("--task or --task-file is required".into()),
        (Some(_), Some(_)) => unreachable!(),
    };
    if task.trim().is_empty() {
        return Err("the task must not be empty".into());
    }

    Ok(Some(Options {
        repo: repo.ok_or_else(|| format!("--repo is required\n\n{}", usage()))?,
        ollama: ollama.ok_or_else(|| format!("--ollama is required\n\n{}", usage()))?,
        opencode,
        models: models.ok_or_else(|| format!("--models is required\n\n{}", usage()))?,
        model: model.ok_or_else(|| format!("--model is required\n\n{}", usage()))?,
        task,
        pairs,
        baseline,
        out,
        timeout: Duration::from_secs(timeout_secs),
        health_timeout: Duration::from_secs(health_timeout_secs),
        context_length,
        seed,
        alpha,
        keep_worktrees,
    }))
}

fn usage() -> &'static str {
    "Usage:
  opencode-oldstring-mask-experiment --repo DIR --ollama PATH --models DIR --model MODEL \\
    (--task TEXT | --task-file FILE) [options]

Required:
  --repo DIR             Clean Git repository used as the common baseline
  --ollama PATH          Ollama binary built with this edit-constraint change
  --models DIR           Ollama model store, e.g. /mnt/sda12/ollama
  --model MODEL          Exact Ollama model name
  --task TEXT            Task prompt; mutually exclusive with --task-file
  --task-file FILE       Read the common task prompt from a UTF-8 file

Options:
  --opencode PATH        OpenCode binary (default: opencode from PATH)
  --n N, --pairs N       Matched pairs; N runs per condition (default: 10)
  --baseline REF         Git commit/ref used for every worktree (default: HEAD)
  --out DIR              Results directory outside the Git repository
  --timeout-secs N       OpenCode run timeout (default: 900)
  --health-timeout-secs N Ollama startup timeout (default: 180)
  --context-length N     Set OLLAMA_CONTEXT_LENGTH for both conditions
  --seed N               Reproducible sampling and balanced order seed (default: 20261106)
  --alpha P              Significance threshold (default: 0.05)
  --keep-worktrees       Keep each trial checkout after saving its patch
  -h, --help             Show this help

Each pair runs the same task against separate worktrees at the same commit.
Control and treatment order alternates from a seeded starting order. The
runner uses the same Ollama and OpenCode binaries in both conditions, changing
only OLLAMA_OPENCODE_EDIT_CONSTRAINTS. It records logs and patches, and reports
an exact two-sided McNemar test for the paired binary outcomes. The task must
exercise OpenCode's read-then-edit flow, and the model must use native
llama.cpp chat for the constraint to apply.
"
}

fn run_trial(
    options: &Options,
    base_commit: &str,
    prompt: &str,
    output: &Path,
    pair: usize,
    condition: Condition,
    order: &'static str,
) -> Result<TrialResult, String> {
    let name = format!("pair-{pair:03}-{}", condition.name());
    let trial_dir = output.join("trials").join(&name);
    // Reuse exactly the same absolute working directory in both conditions.
    // Archive the dirty checkout after each run so OpenCode sees identical paths.
    let worktree = output.join("worktrees").join(format!("pair-{pair:03}-workspace"));
    let archive = output.join("worktrees").join(&name);
    fs::create_dir_all(&trial_dir).map_err(display_error)?;
    write_text(&trial_dir.join("prompt.txt"), prompt)?;

    git_status(
        &options.repo,
        &[
            "worktree",
            "add",
            "--detach",
            worktree.to_str().ok_or("worktree path is not UTF-8")?,
            base_commit,
        ],
    )?;

    let trial_result = run_trial_in_worktree(
        options, prompt, &trial_dir, &worktree, pair, condition, order,
    );
    let patch_path = trial_dir.join("changes.patch");
    let status_path = trial_dir.join("git-status.txt");
    let capture_result = capture_worktree_diff(&worktree, &patch_path, &status_path);
    let cleanup_result = if options.keep_worktrees {
        git_status(&options.repo, &["worktree", "move",
            worktree.to_str().ok_or("worktree path is not UTF-8")?,
            archive.to_str().ok_or("archive path is not UTF-8")?])
    } else if let Some(path) = worktree.to_str() {
        git_status(&options.repo, &["worktree", "remove", "--force", path])
    } else {
        Err("worktree path is not UTF-8".to_owned())
    };

    let mut result = trial_result?;
    capture_result?;
    cleanup_result?;
    result.patch_path = patch_path;
    Ok(result)
}

fn run_trial_in_worktree(
    options: &Options,
    prompt: &str,
    trial_dir: &Path,
    worktree: &Path,
    pair: usize,
    condition: Condition,
    order: &'static str,
) -> Result<TrialResult, String> {
    let ollama_log = trial_dir.join("ollama.log");
    let mut server = start_ollama(options, condition, &ollama_log)?;
    verify_model_available(options, server.port)?;

    let profile_dir = trial_dir.join("opencode-profile");
    let config_dir = profile_dir.join("config");
    let data_dir = profile_dir.join("data");
    let cache_dir = profile_dir.join("cache");
    for dir in [&config_dir, &data_dir, &cache_dir] {
        fs::create_dir_all(dir).map_err(display_error)?;
    }
    let config = opencode_config(&options.model, server.port, options.seed);
    write_text(&trial_dir.join("opencode-config.json"), &config)?;

    let stdout_path = trial_dir.join("opencode.stdout.jsonl");
    let stderr_path = trial_dir.join("opencode.stderr.log");
    let stdout = File::create(&stdout_path).map_err(display_error)?;
    let stderr = File::create(&stderr_path).map_err(display_error)?;
    let run_title = format!("sampling experiment {pair} {}", condition.name());
    let model_id = format!("{PROVIDER_ID}/{}", options.model);
    let mut command = Command::new(&options.opencode);
    command
        .arg("--print-logs")
        .arg("run")
        .arg("--model")
        .arg(model_id)
        .arg("--auto")
        .arg("--pure")
        .arg("--format")
        .arg("json")
        .arg("--title")
        .arg(run_title)
        .arg(prompt)
        .current_dir(worktree)
        // OpenCode uses PWD as the session root when present, so keep it in
        // sync with current_dir instead of inheriting the experiment runner's
        // workspace directory.
        .env("PWD", worktree)
        .env("OPENCODE_CONFIG_CONTENT", &config)
        .env("OPENCODE_CONFIG_DIR", &config_dir)
        .env("OPENCODE_EXPERIMENT_OUTPUT", trial_dir)
        .env_remove("OPENCODE_CONFIG")
        .env_remove("OPENCODE_CONFIG_PROJECT_DISABLE")
        .env_remove("OPENCODE_DISABLE_PROJECT_CONFIG")
        .env("XDG_CONFIG_HOME", profile_dir.join("xdg-config"))
        .env("XDG_DATA_HOME", &data_dir)
        .env("XDG_CACHE_HOME", &cache_dir)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    command.process_group(0);

    let started = Instant::now();
    let mut child = command
        .spawn()
        .map_err(|error| format!("cannot start OpenCode: {error}"))?;
    let wait_result = wait_for_opencode(&mut child, &mut server, options.timeout);
    stop_process_group(&mut child);
    let (status, timed_out) = wait_result?;
    let elapsed_secs = started.elapsed().as_secs_f64();
    drop(server);

    let events = parse_opencode_events(&stdout_path)?;
    let success_marker = has_exact_line(&events.final_text_after_last_tool, SUCCESS_MARKER);
    let failure_marker = has_exact_line(&events.final_text_after_last_tool, FAILURE_MARKER);
    let self_report = match (success_marker, failure_marker) {
        (true, false) => "success",
        (false, true) => "failure",
        (true, true) => "ambiguous",
        (false, false) => "missing",
    };
    let exit_code = status
        .as_ref()
        .and_then(ExitStatus::code)
        .map(|code| code.to_string())
        .unwrap_or_else(|| "signal-or-unknown".to_owned());
    let clean_exit = !timed_out
        && status.as_ref().map(ExitStatus::success).unwrap_or(false)
        && events.error_events == 0
        && events.open_steps == 0;
    let success = clean_exit && success_marker && !failure_marker;
    let reason = if timed_out {
        "timeout".to_owned()
    } else if !status.as_ref().map(ExitStatus::success).unwrap_or(false) {
        format!("OpenCode exited with {exit_code}")
    } else if events.error_events > 0 {
        format!("{} OpenCode error event(s)", events.error_events)
    } else if events.open_steps > 0 {
        format!("{} unfinished OpenCode step(s)", events.open_steps)
    } else if failure_marker {
        "self-reported failure".to_owned()
    } else if !success_marker {
        "missing final success marker".to_owned()
    } else {
        "clean exit with final success marker".to_owned()
    };

    Ok(TrialResult {
        pair,
        condition,
        order,
        success,
        self_report,
        reason,
        elapsed_secs,
        exit_code,
        timed_out,
        tool_errors: events.tool_errors,
        error_events: events.error_events,
        open_steps: events.open_steps,
        stdout_path,
        stderr_path,
        patch_path: PathBuf::new(),
    })
}

fn start_ollama(
    options: &Options,
    condition: Condition,
    log_path: &Path,
) -> Result<RunningOllama, String> {
    let listener = TcpListener::bind(("127.0.0.1", 0)).map_err(display_error)?;
    let port = listener.local_addr().map_err(display_error)?.port();
    drop(listener);
    let host = format!("127.0.0.1:{port}");

    let log = File::create(log_path).map_err(display_error)?;
    let log_copy = log.try_clone().map_err(display_error)?;
    let mut command = Command::new(&options.ollama);
    command
        .arg("serve")
        .env("OLLAMA_HOST", &host)
        .env("OLLAMA_MODELS", &options.models)
        .env("OLLAMA_OPENCODE_EDIT_CONSTRAINTS", condition.flag_value())
        .env("OLLAMA_OPENCODE_EDIT_EXPERIMENT", "1")
        .env("OLLAMA_NUM_PARALLEL", "1")
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_copy));
    if let Some(context_length) = options.context_length {
        command.env("OLLAMA_CONTEXT_LENGTH", context_length.to_string());
    }
    command.process_group(0);
    let child = command
        .spawn()
        .map_err(|error| format!("cannot start Ollama: {error}"))?;
    let mut server = RunningOllama {
        child,
        port,
        log_path: log_path.to_path_buf(),
    };

    let deadline = Instant::now() + options.health_timeout;
    loop {
        if let Some(status) = server.child.try_wait().map_err(display_error)? {
            return Err(format!(
                "Ollama exited during startup with {status}; see {}",
                server.log_path.display()
            ));
        }
        if probe_ollama(port) {
            return Ok(server);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "Ollama did not become ready within {} seconds; see {}",
                options.health_timeout.as_secs(),
                server.log_path.display()
            ));
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn verify_model_available(options: &Options, port: u16) -> Result<(), String> {
    let host = format!("127.0.0.1:{port}");
    let output = Command::new(&options.ollama)
        .arg("show")
        .arg(&options.model)
        .env("OLLAMA_HOST", &host)
        .env("OLLAMA_MODELS", &options.models)
        .output()
        .map_err(|error| format!("cannot check model availability: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "model {:?} is not available from the isolated Ollama server: {}\nCheck --model and --models.",
            options.model,
            stderr.trim()
        ));
    }
    Ok(())
}

fn probe_ollama(port: u16) -> bool {
    let address = format!("127.0.0.1:{port}");
    let Ok(mut stream) =
        TcpStream::connect_timeout(&address.parse().unwrap(), Duration::from_millis(400))
    else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(700)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(700)));
    if stream
        .write_all(b"GET /api/version HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut response = [0u8; 64];
    match stream.read(&mut response) {
        Ok(count) => {
            response[..count].starts_with(b"HTTP/1.1 200")
                || response[..count].starts_with(b"HTTP/1.0 200")
        }
        Err(_) => false,
    }
}

fn wait_for_opencode(
    child: &mut Child,
    server: &mut RunningOllama,
    timeout: Duration,
) -> Result<(Option<ExitStatus>, bool), String> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().map_err(display_error)? {
            return Ok((Some(status), false));
        }
        if let Some(status) = server.child.try_wait().map_err(display_error)? {
            stop_process_group(child);
            return Err(format!(
                "Ollama exited while OpenCode was running with {status}; see {}",
                server.log_path.display()
            ));
        }
        if started.elapsed() >= timeout {
            stop_process_group(child);
            return Ok((None, true));
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn stop_process_group(child: &mut Child) {
    let group = child.id() as i32;
    unsafe {
        let _ = kill(-group, 15);
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if child.try_wait().ok().flatten().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }
    unsafe {
        let _ = kill(-group, 9);
    }
    let _ = child.wait();
}

fn opencode_config(model: &str, port: u16, seed: u64) -> String {
    let model_json = json_quote(model);
    let base_url = json_quote(&format!("http://127.0.0.1:{port}/v1"));
    let mut config = String::from(
        "{\n  \"$schema\": \"https://opencode.ai/config.json\",\n  \"provider\": {\n    ",
    );
    config.push_str(&json_quote(PROVIDER_ID));
    config.push_str(
        ": {\n      \"npm\": \"@ai-sdk/openai-compatible\",\n\
         \"name\": \"Ollama experiment\",\n\
         \"options\": {\n        \"baseURL\": ",
    );
    config.push_str(&base_url);
    config.push_str(
        ",\n        \"apiKey\": \"ollama\",\n        \"headerTimeout\": false\n      },\n\
         \"models\": {\n        ",
    );
    config.push_str(&model_json);
    config.push_str(": { \"name\": ");
    config.push_str(&model_json);
    config.push_str(", \"options\": { \"think\": false, \"seed\": ");
    config.push_str(&seed.to_string());
    config.push_str(" } }\n      }\n    }\n  }\n}\n");
    config
}

fn scored_prompt(task: &str) -> String {
    format!(
        "{task}\n\n\
         After attempting the task, end your final response with exactly one standalone line:\n\
         {SUCCESS_MARKER}\n\
         if you believe you completed the task, or:\n\
         {FAILURE_MARKER}\n\
         if you did not complete it. Do not claim success if you are uncertain."
    )
}

fn parse_opencode_events(path: &Path) -> Result<EventSummary, String> {
    let file = File::open(path).map_err(display_error)?;
    let mut summary = EventSummary {
        final_text_after_last_tool: String::new(),
        error_events: 0,
        tool_errors: 0,
        open_steps: 0,
    };
    use std::io::{BufRead, BufReader};
    for line in BufReader::new(file).lines() {
        let line = line.map_err(display_error)?;
        let Some(event_type) = json_string_field(&line, "type", 0) else {
            continue;
        };
        match event_type.as_str() {
            "text" => {
                let part_start = object_field_value_start(&line, "part", 0).unwrap_or(0);
                if let Some(text) = json_string_field(&line, "text", part_start) {
                    summary.final_text_after_last_tool.push_str(&text);
                    summary.final_text_after_last_tool.push('\n');
                }
            }
            "tool_use" => {
                let part_start = object_field_value_start(&line, "part", 0).unwrap_or(0);
                let state_start =
                    object_field_value_start(&line, "state", part_start).unwrap_or(part_start);
                if json_string_field(&line, "status", state_start).as_deref() == Some("error") {
                    summary.tool_errors += 1;
                }
                summary.final_text_after_last_tool.clear();
            }
            "error" => summary.error_events += 1,
            "step_start" => summary.open_steps += 1,
            "step_finish" => summary.open_steps = summary.open_steps.saturating_sub(1),
            _ => {}
        }
    }
    Ok(summary)
}

fn object_field_value_start(input: &str, key: &str, from: usize) -> Option<usize> {
    let bytes = input.as_bytes();
    let marker = format!("\"{key}\"").into_bytes();
    let mut index = from.min(bytes.len());
    while index + marker.len() <= bytes.len() {
        if bytes[index..index + marker.len()] == marker {
            let mut cursor = index + marker.len();
            skip_space(bytes, &mut cursor);
            if bytes.get(cursor) != Some(&b':') {
                index += 1;
                continue;
            }
            cursor += 1;
            skip_space(bytes, &mut cursor);
            if bytes.get(cursor) == Some(&b'{') {
                return Some(cursor);
            }
            return None;
        }
        index += 1;
    }
    None
}

fn json_string_field(input: &str, key: &str, from: usize) -> Option<String> {
    let bytes = input.as_bytes();
    let marker = format!("\"{key}\"").into_bytes();
    let mut index = from.min(bytes.len());
    while index + marker.len() <= bytes.len() {
        if bytes[index..index + marker.len()] != marker {
            index += 1;
            continue;
        }
        let mut cursor = index + marker.len();
        skip_space(bytes, &mut cursor);
        if bytes.get(cursor) != Some(&b':') {
            index += 1;
            continue;
        }
        cursor += 1;
        skip_space(bytes, &mut cursor);
        if bytes.get(cursor) != Some(&b'"') {
            index += 1;
            continue;
        }
        return decode_json_string(bytes, cursor + 1).map(|parsed| parsed.0);
    }
    None
}

fn skip_space(bytes: &[u8], cursor: &mut usize) {
    while matches!(
        bytes.get(*cursor).copied(),
        Some(b' ' | b'\n' | b'\r' | b'\t')
    ) {
        *cursor += 1;
    }
}

fn decode_json_string(bytes: &[u8], mut cursor: usize) -> Option<(String, usize)> {
    let mut output = Vec::new();
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'"' => {
                return String::from_utf8(output)
                    .ok()
                    .map(|text| (text, cursor + 1))
            }
            b'\\' => {
                cursor += 1;
                match *bytes.get(cursor)? {
                    b'"' => output.push(b'"'),
                    b'\\' => output.push(b'\\'),
                    b'/' => output.push(b'/'),
                    b'b' => output.push(8),
                    b'f' => output.push(12),
                    b'n' => output.push(b'\n'),
                    b'r' => output.push(b'\r'),
                    b't' => output.push(b'\t'),
                    b'u' => {
                        cursor += 1;
                        let first = parse_hex4(bytes.get(cursor..cursor + 4)?)?;
                        cursor += 4;
                        let scalar = if (0xd800..=0xdbff).contains(&first) {
                            if bytes.get(cursor..cursor + 2)? != b"\\u" {
                                return None;
                            }
                            let second = parse_hex4(bytes.get(cursor + 2..cursor + 6)?)?;
                            if !(0xdc00..=0xdfff).contains(&second) {
                                return None;
                            }
                            cursor += 6;
                            0x10000 + (((first - 0xd800) as u32) << 10) + (second - 0xdc00) as u32
                        } else {
                            first as u32
                        };
                        let character = char::from_u32(scalar)?;
                        let mut encoded = [0u8; 4];
                        output.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
                        cursor -= 1;
                    }
                    _ => return None,
                }
            }
            byte if byte < 0x20 => return None,
            byte if byte < 0x80 => output.push(byte),
            _ => {
                let width = match bytes[cursor] {
                    0xc2..=0xdf => 2,
                    0xe0..=0xef => 3,
                    0xf0..=0xf4 => 4,
                    _ => return None,
                };
                let slice = bytes.get(cursor..cursor + width)?;
                std::str::from_utf8(slice).ok()?;
                output.extend_from_slice(slice);
                cursor += width - 1;
            }
        }
        cursor += 1;
    }
    None
}

fn parse_hex4(bytes: &[u8]) -> Option<u16> {
    if bytes.len() != 4 {
        return None;
    }
    let mut value = 0u16;
    for byte in bytes {
        value = value.checked_mul(16)?;
        value += match byte {
            b'0'..=b'9' => (byte - b'0') as u16,
            b'a'..=b'f' => (byte - b'a' + 10) as u16,
            b'A'..=b'F' => (byte - b'A' + 10) as u16,
            _ => return None,
        };
    }
    Some(value)
}

fn has_exact_line(text: &str, expected: &str) -> bool {
    text.lines().any(|line| line.trim() == expected)
}

fn capture_worktree_diff(
    worktree: &Path,
    patch_path: &Path,
    status_path: &Path,
) -> Result<(), String> {
    git_status(worktree, &["add", "--intent-to-add", "--all"])?;
    let diff = Command::new("git")
        .args(["diff", "--binary", "--no-ext-diff", "--"])
        .current_dir(worktree)
        .output()
        .map_err(display_error)?;
    if !diff.status.success() {
        return Err(format!(
            "cannot capture trial diff: {}",
            String::from_utf8_lossy(&diff.stderr)
        ));
    }
    fs::write(patch_path, diff.stdout).map_err(display_error)?;

    let status = git_capture(worktree, &["status", "--short", "--untracked-files=all"])?;
    write_text(status_path, &status)?;
    Ok(())
}

fn append_csv(
    path: &Path,
    result: &TrialResult,
    base_commit: &str,
    model: &str,
) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(display_error)?;
    let fields = [
        result.pair.to_string(),
        result.condition.name().to_owned(),
        result.order.to_owned(),
        base_commit.to_owned(),
        model.to_owned(),
        format!("{:.3}", result.elapsed_secs),
        result.exit_code.clone(),
        result.timed_out.to_string(),
        result.self_report.to_owned(),
        result.success.to_string(),
        result.reason.clone(),
        result.tool_errors.to_string(),
        result.error_events.to_string(),
        result.open_steps.to_string(),
        result.stdout_path.display().to_string(),
        result.stderr_path.display().to_string(),
        result.patch_path.display().to_string(),
    ];
    writeln!(
        file,
        "{}",
        fields
            .iter()
            .map(|field| csv_quote(field))
            .collect::<Vec<_>>()
            .join(",")
    )
    .map_err(display_error)
}

fn summarize(results: &[TrialResult], pairs: usize, alpha: f64) -> String {
    let mut control_successes = 0usize;
    let mut constrained_successes = 0usize;
    let mut control_only = 0usize;
    let mut constrained_only = 0usize;

    for pair in 1..=pairs {
        let control = results
            .iter()
            .find(|result| result.pair == pair && result.condition == Condition::Control)
            .map(|result| result.success)
            .unwrap_or(false);
        let constrained = results
            .iter()
            .find(|result| result.pair == pair && result.condition == Condition::Constrained)
            .map(|result| result.success)
            .unwrap_or(false);
        control_successes += usize::from(control);
        constrained_successes += usize::from(constrained);
        match (control, constrained) {
            (true, false) => control_only += 1,
            (false, true) => constrained_only += 1,
            _ => {}
        }
    }

    let p_value = mcnemar_exact_two_sided(control_only, constrained_only);
    let control_rate = control_successes as f64 / pairs as f64;
    let constrained_rate = constrained_successes as f64 / pairs as f64;
    let significant = p_value < alpha;
    format!(
        "Results\n\
         Pairs: {pairs} matched pairs ({} runs per condition; {} runs total)\n\
         Control: {control_successes}/{pairs} ({:.1}%)\n\
         Constrained: {constrained_successes}/{pairs} ({:.1}%)\n\
         Paired discordances: {constrained_only} favor constrained, {control_only} favor control\n\
         Success-rate difference (constrained minus control): {:+.1} percentage points\n\
         Exact two-sided McNemar p-value: {:.6}\n\
         Alpha: {alpha:.3}; statistically significant: {}\n\
         Scoring requires a final standalone self-report marker and a clean OpenCode exit.\n\
         The treatment applies to native llama.cpp chat requests with an OpenCode working directory and edit tool.\n\
         The CSV score uses self-report; actual edits are scored separately in actual-edit-score.json. These repeated runs estimate this task, model, and baseline; they do not establish an effect across other tasks or repositories.\n",
        pairs,
        pairs * 2,
        control_rate * 100.0,
        constrained_rate * 100.0,
        (constrained_rate - control_rate) * 100.0,
        p_value,
        if significant { "yes" } else { "no" },
    )
}

fn mcnemar_exact_two_sided(control_only: usize, constrained_only: usize) -> f64 {
    let discordant = control_only + constrained_only;
    if discordant == 0 {
        return 1.0;
    }
    let k = control_only.min(constrained_only);
    let mut log_factorial = vec![0.0f64; discordant + 1];
    for value in 1..=discordant {
        log_factorial[value] = log_factorial[value - 1] + (value as f64).ln();
    }
    let log_two_power = discordant as f64 * 2.0f64.ln();
    let logs: Vec<f64> = (0..=k)
        .map(|value| {
            log_factorial[discordant]
                - log_factorial[value]
                - log_factorial[discordant - value]
                - log_two_power
        })
        .collect();
    let max_log = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let tail = max_log.exp()
        * logs
            .iter()
            .map(|value| (value - max_log).exp())
            .sum::<f64>();
    (2.0 * tail).min(1.0)
}

fn command_version(program: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("cannot run {}: {error}", program.display()))?;
    if !output.status.success() {
        return Err(format!(
            "{} {} failed: {}",
            program.display(),
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    Ok(format!("{stdout}{stderr}").trim().to_owned())
}

fn git_capture(repo: &Path, args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .map_err(display_error)?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed in {}: {}",
            args.join(" "),
            repo.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn git_status(repo: &Path, args: &[&str]) -> Result<(), String> {
    let _ = git_capture(repo, args)?;
    Ok(())
}

fn resolve_program(path: &Path) -> Result<PathBuf, String> {
    if path.components().count() > 1 || path.is_absolute() {
        return canonicalize(path);
    }
    Ok(path.to_path_buf())
}

fn canonicalize(path: &Path) -> Result<PathBuf, String> {
    fs::canonicalize(path).map_err(|error| format!("{}: {error}", path.display()))
}

fn absolute_path(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir().map_err(display_error)?.join(path)
    };
    let mut ancestor = absolute.clone();
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        let name = ancestor
            .file_name()
            .ok_or_else(|| format!("invalid path: {}", absolute.display()))?;
        suffix.push(name.to_os_string());
        if !ancestor.pop() {
            return Err(format!("invalid path: {}", absolute.display()));
        }
    }
    let mut resolved = canonicalize(&ancestor)?;
    for component in suffix.iter().rev() {
        resolved.push(component);
    }
    Ok(resolved)
}

fn write_text(path: &Path, text: &str) -> Result<(), String> {
    fs::write(path, text).map_err(|error| format!("{}: {error}", path.display()))
}

fn display_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn json_quote(input: &str) -> String {
    let mut output = String::with_capacity(input.len() + 2);
    output.push('"');
    for character in input.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character < '\u{20}' => {
                output.push_str(&format!("\\u{:04x}", character as u32))
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn csv_quote(input: &str) -> String {
    format!("\"{}\"", input.replace('"', "\"\""))
}

fn next_random(state: &mut u64) -> u64 {
    let mut value = *state;
    if value == 0 {
        value = 0x9e3779b97f4a7c15;
    }
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value
}
