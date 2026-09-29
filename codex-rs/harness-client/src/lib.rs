//! stdio JSON-RPC client for a deepseek-harness runtime capability provider.
//!
//! Owns one child process (`dsh --profile <profile>`) with piped stdin/stdout,
//! newline-delimited JSON-RPC framing shared with the DSH SDK transport, and
//! id-based request correlation supporting concurrent in-flight requests.
//!
//! Wire dialect: outbound frames omit the `"jsonrpc"` field (the Codex JSON-RPC
//! dialect used by exec-server, `codex-exec-server-protocol/src/rpc.rs`); the
//! DSH `JsonRpcLineTransport` dispatches on the presence of `id`/`method` only
//! and accepts both dialects. Inbound frames carrying `"jsonrpc"` deserialize
//! fine because the shared wire structs do not deny unknown fields.
//!
//! Lifecycle: the client is owned by [`HarnessRuntimeManager`] on
//! `SessionServices` (per-thread, session-level). Tool handlers only hold an
//! `Arc` clone; they never spawn or tear down the runtime.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use codex_exec_server_protocol::JSONRPCMessage;
use codex_exec_server_protocol::JSONRPCRequest;
use codex_exec_server_protocol::RequestId;
use serde_json::Value;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::process::Child;
use tokio::process::ChildStdin;
use tokio::process::ChildStdout;
use tokio::sync::Mutex;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, HarnessRequestError>>>>>;

/// Error surfaced by [`HarnessClient::request`].
#[derive(Debug)]
pub enum HarnessRequestError {
    /// The runtime is not running (disabled, failed to spawn, or stopped).
    Unavailable(&'static str),
    /// The runtime replied with a JSON-RPC error frame.
    Remote { code: i64, message: String },
    /// Write failure, unexpected child exit, or request timeout.
    Transport(String),
}

impl std::fmt::Display for HarnessRequestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(reason) => write!(f, "harness runtime unavailable: {reason}"),
            Self::Remote { code, message } => write!(f, "harness runtime error {code}: {message}"),
            Self::Transport(message) => write!(f, "harness transport error: {message}"),
        }
    }
}

impl std::error::Error for HarnessRequestError {}

/// Environment variable pointing at the DSH CLI entry (JS entry or bin path
/// from the DSH repo's `apps/cli`). This is a PoC prerequisite, not a config
/// surface: sessions with `harness.enabled` fail to spawn (and degrade to the
/// disabled manager) when it is unset.
pub const CODEX_HARNESS_DSH_BIN_ENV: &str = "CODEX_HARNESS_DSH_BIN";

/// Environment variable overriding the node executable used to launch the DSH
/// runtime. Implementation detail: unset means `node` resolved via `PATH`.
pub const CODEX_HARNESS_NODE_ENV: &str = "CODEX_HARNESS_NODE";

/// Resolve the fixed PoC launch contract argv for the DSH runtime:
/// `<node> <dsh-entry> --profile <profile>`.
///
/// This is deliberately not configurable beyond the profile name — config must
/// not become an arbitrary process launcher (SPEC §10). The profile name is
/// validated so a config value can never masquerade as an extra flag.
pub fn launch_argv(profile: &str) -> std::io::Result<Vec<std::ffi::OsString>> {
    if profile.is_empty()
        || profile.starts_with('-')
        || profile
            .chars()
            .any(|c| c.is_whitespace() || c == '/' || c == '\\' || c.is_control())
    {
        return Err(std::io::Error::other(format!(
            "invalid harness profile name: {profile:?}"
        )));
    }
    let node = std::env::var(CODEX_HARNESS_NODE_ENV).unwrap_or_else(|_| "node".to_string());
    let dsh_bin = std::env::var(CODEX_HARNESS_DSH_BIN_ENV).map_err(|_| {
        std::io::Error::other(format!(
            "`{CODEX_HARNESS_DSH_BIN_ENV}` is not set; point it at the DSH CLI entry \
             (apps/cli JS entry or bin path) — a PoC prerequisite for the harness runtime"
        ))
    })?;
    Ok(vec![
        node.into(),
        dsh_bin.into(),
        "--profile".into(),
        profile.into(),
    ])
}

/// Handle to one spawned DSH runtime capability provider.
pub struct HarnessClient {
    next_id: AtomicU64,
    pending: PendingMap,
    /// `None` after shutdown: dropping the sender closes the child's stdin,
    /// which is the runtime's documented EOF shutdown trigger.
    stdin_tx: Mutex<Option<mpsc::Sender<String>>>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
    /// `None` after shutdown has taken the child for its exit/kill ladder.
    child: Mutex<Option<Child>>,
}

impl HarnessClient {
    /// Spawn the DSH runtime and start the reader/writer tasks.
    ///
    /// `argv` is the fixed launch contract (node + dsh entry + `--profile`),
    /// resolved by the caller from config; the client never invents arguments.
    pub async fn spawn(argv: &[std::ffi::OsString], cwd: &Path) -> std::io::Result<Arc<Self>> {
        let (program, args) = argv
            .split_first()
            .ok_or_else(|| std::io::Error::other("harness runtime argv must not be empty"))?;
        let mut command = tokio::process::Command::new(program);
        command
            .args(args)
            .current_dir(cwd)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            // Discard the child's stderr: stdout is the only protocol channel
            // and the runtime routes its diagnostics to an in-memory logger.
            // A piped stderr nobody reads would fill after 64 KiB and block
            // the child forever (the same trade-off as the exec-server and
            // rmcp-client child spawns).
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true);
        let mut child = command.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("harness runtime stdin was not piped"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("harness runtime stdout was not piped"))?;

        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let (stdin_tx, stdin_rx) = mpsc::channel::<String>(64);

        let reader = tokio::spawn(read_frames(stdout, Arc::clone(&pending)));
        let writer = tokio::spawn(write_frames(stdin, stdin_rx));

        Ok(Arc::new(Self {
            next_id: AtomicU64::new(1),
            pending,
            stdin_tx: Mutex::new(Some(stdin_tx)),
            tasks: Mutex::new(vec![reader, writer]),
            child: Mutex::new(Some(child)),
        }))
    }

    /// Send one JSON-RPC request and await the correlated response.
    ///
    /// Concurrent callers are safe: correlation is by monotonically assigned
    /// request id, mirroring the DSH transport's pending map.
    pub async fn request(&self, method: &str, params: Value) -> Result<Value, HarnessRequestError> {
        let Some(stdin_tx) = self.stdin_tx.lock().await.clone() else {
            return Err(HarnessRequestError::Unavailable(
                "harness runtime is shut down",
            ));
        };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let message = JSONRPCMessage::Request(JSONRPCRequest {
            id: RequestId::Integer(i64::try_from(id).unwrap_or(i64::MAX)),
            method: method.to_string(),
            params: Some(params),
            trace: None,
        });
        let frame = serde_json::to_string(&message)
            .map_err(|error| HarnessRequestError::Transport(error.to_string()))?;

        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id, tx);
        if stdin_tx.send(frame).await.is_err() {
            // The writer task is gone: drop the pending entry (no leaked map
            // slot) and fail this request.
            self.pending.lock().await.remove(&id);
            return Err(HarnessRequestError::Unavailable(
                "writer task stopped (runtime exited)",
            ));
        }

        rx.await.unwrap_or(Err(HarnessRequestError::Unavailable(
            "request dropped before response (runtime exited or was shut down)",
        )))
    }

    /// Shut the runtime down: close stdin (EOF → the runtime disposes and
    /// exits 0), await exit, then kill after a grace period. Mirrors the DSH
    /// EOF → SIGTERM → SIGKILL ladder.
    ///
    /// Returns `Err` when the runtime failed to exit gracefully within the
    /// grace period (it is killed regardless) or when polling the child failed.
    pub async fn shutdown(&self) -> std::io::Result<()> {
        // Fail every in-flight request first: aborting the reader task below
        // skips its natural end-of-stdout drain, so without this step pending
        // callers would hang until the client is dropped (failPending mirror of
        // the DSH transport's close()).
        {
            let mut pending = self.pending.lock().await;
            for (_, sender) in pending.drain() {
                let _ = sender.send(Err(HarnessRequestError::Unavailable(
                    "harness runtime is shutting down",
                )));
            }
        }
        // Drop the stdin sender; write_frames then drains, closes stdin (EOF),
        // and exits once the runtime stops sending frames.
        self.stdin_tx.lock().await.take();
        // Collect the task handles inside a tight scope: the guard must not be
        // held across the awaits below (clippy::await_holding_invalid_type).
        let tasks: Vec<JoinHandle<()>> = {
            let mut guard = self.tasks.lock().await;
            guard.drain(..).collect()
        };
        for task in tasks {
            task.abort();
        }
        // Take the child out of the lock so the exit/kill ladder awaits without
        // holding the guard (shutdown is terminal; the child is not put back).
        let Some(mut child) = self.child.lock().await.take() else {
            return Ok(());
        };
        for _ in 0..50 {
            match child.try_wait() {
                Ok(Some(_)) => return Ok(()),
                Ok(None) => tokio::time::sleep(std::time::Duration::from_millis(100)).await,
                Err(error) => return Err(error),
            }
        }
        let _ = child.start_kill();
        let _ = child.wait().await;
        Err(std::io::Error::other(
            "harness runtime did not exit within the grace period; killed",
        ))
    }
}

/// Read newline-delimited JSON-RPC frames and correlate responses by id.
async fn read_frames(stdout: ChildStdout, pending: PendingMap) {
    let reader = BufReader::new(stdout);
    let mut lines = reader.lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if line.trim().is_empty() {
            continue;
        }
        let decoded = serde_json::from_str::<JSONRPCMessage>(&line);
        let resolved = match decoded {
            // DSH result frame: {"jsonrpc":"2.0","id":N,"result":...}
            Ok(JSONRPCMessage::Response(response)) => {
                let Some(id) = request_id(&response.id) else {
                    continue;
                };
                Some((id, Ok(response.result)))
            }
            // DSH error frame: {"jsonrpc":"2.0","id":N,"error":{code,message}}
            Ok(JSONRPCMessage::Error(error)) => {
                let Some(id) = request_id(&error.id) else {
                    continue;
                };
                Some((
                    id,
                    Err(HarnessRequestError::Remote {
                        code: error.error.code,
                        message: error.error.message,
                    }),
                ))
            }
            // Requests/notifications from the runtime are not part of the PoC
            // method space; malformed lines are ignored exactly as the DSH
            // transport ignores them (transport.ts:202-208).
            other => {
                tracing::debug!(line = %line, decoded_ok = other.is_ok(), "ignoring harness frame");
                continue;
            }
        };
        let Some((id, outcome)) = resolved else {
            continue;
        };
        if let Some(sender) = pending.lock().await.remove(&id) {
            let _ = sender.send(outcome);
        }
    }
    // Runtime stdout closed: fail everything still pending.
    let mut map = pending.lock().await;
    for (_, sender) in map.drain() {
        let _ = sender.send(Err(HarnessRequestError::Unavailable(
            "harness runtime closed stdout before responding",
        )));
    }
}

fn request_id(id: &RequestId) -> Option<u64> {
    match id {
        RequestId::Integer(value) => u64::try_from(*value).ok(),
        RequestId::String(_) => None,
    }
}

async fn write_frames(mut stdin: ChildStdin, mut rx: mpsc::Receiver<String>) {
    while let Some(frame) = rx.recv().await {
        if stdin.write_all(frame.as_bytes()).await.is_err()
            || stdin.write_all(b"\n").await.is_err()
            || stdin.flush().await.is_err()
        {
            break;
        }
    }
    // Dropping `stdin` signals EOF to the runtime.
}

/// Session-owned owner of the DSH runtime process (r2 SPEC §3.1).
///
/// Lives on `SessionServices`; created once per thread when
/// `config.harness.enabled`, torn down in `shutdown_session_runtime`. Tool
/// handlers receive `Arc` clones and never spawn or kill the runtime.
pub enum HarnessRuntimeManager {
    /// Config disabled or startup failed; tool registration omits the tools.
    Disabled,
    Running(Arc<HarnessClient>),
}

impl HarnessRuntimeManager {
    pub fn disabled() -> Self {
        Self::Disabled
    }

    pub fn is_available(&self) -> bool {
        matches!(self, Self::Running(_))
    }

    pub fn client(&self) -> Option<Arc<HarnessClient>> {
        match self {
            Self::Disabled => None,
            Self::Running(client) => Some(Arc::clone(client)),
        }
    }

    /// Send a capability request; `Unavailable` maps to the codex-side
    /// runtime-unavailable tool error (SPEC §5: never sent on the wire).
    pub async fn request(&self, method: &str, params: Value) -> Result<Value, HarnessRequestError> {
        match self {
            Self::Disabled => Err(HarnessRequestError::Unavailable("harness is disabled")),
            Self::Running(client) => client.request(method, params).await,
        }
    }

    /// Tear the runtime down. `Ok(())` on a graceful exit; `Err` when the
    /// runtime had to be killed after the grace period (session shutdown logs
    /// this as a warning, it never fails the session).
    pub async fn shutdown(&self) -> std::io::Result<()> {
        match self {
            Self::Disabled => Ok(()),
            Self::Running(client) => client.shutdown().await,
        }
    }
}

#[cfg(test)]
mod launch_argv_tests {
    use super::*;

    #[test]
    fn launch_argv_builds_the_fixed_contract_for_a_valid_profile() {
        // SAFETY: single-threaded test scope; the two variables are only read
        // by this test and `launch_argv`.
        unsafe {
            std::env::set_var(CODEX_HARNESS_NODE_ENV, "node-test");
            std::env::set_var(CODEX_HARNESS_DSH_BIN_ENV, "/tmp/dsh-entry.js");
        }
        let argv = launch_argv("harness-capability").expect("valid profile");
        assert_eq!(
            argv,
            vec![
                std::ffi::OsString::from("node-test"),
                std::ffi::OsString::from("/tmp/dsh-entry.js"),
                std::ffi::OsString::from("--profile"),
                std::ffi::OsString::from("harness-capability"),
            ]
        );
    }

    #[test]
    fn launch_argv_rejects_profiles_that_could_masquerade_as_flags() {
        // SAFETY: same single-threaded scope as above; rejection happens before
        // the environment is read.
        unsafe {
            std::env::set_var(CODEX_HARNESS_DSH_BIN_ENV, "/tmp/dsh-entry.js");
        }
        for bad in ["", "--help", "-profile", "a b", "a/b", "a\\b", "a\nb"] {
            assert!(
                launch_argv(bad).is_err(),
                "profile {bad:?} must be rejected"
            );
        }
    }
}
