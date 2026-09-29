//! End-to-end test: real DSH runtime ↔ `HarnessRuntimeManager`.
//!
//! Requires a locally built DSH checkout (pnpm build in deepseek-harness) and
//! is therefore env-gated: set `CODEX_HARNESS_E2E_DSH=1` plus
//! `CODEX_HARNESS_DSH_BIN` (the DSH CLI entry JS) to run:
//!
//! ```text
//! CODEX_HARNESS_E2E_DSH=1 \
//! CODEX_HARNESS_DSH_BIN=/path/to/deepseek-harness/apps/cli/lib/bin.js \
//!   cargo test -p codex-harness-client --test e2e_dsh -- --ignored
//! ```

use std::sync::Arc;

use codex_harness_client::HarnessClient;
use codex_harness_client::HarnessRuntimeManager;
use serde_json::json;

fn dsh_bin() -> String {
    std::env::var("CODEX_HARNESS_DSH_BIN").expect("CODEX_HARNESS_DSH_BIN must point at the DSH CLI entry")
}

/// Spawn the real runtime and run the handshake.
async fn spawn_runtime() -> Arc<HarnessClient> {
    let argv = vec![
        std::ffi::OsString::from("node"),
        std::ffi::OsString::from(dsh_bin()),
        std::ffi::OsString::from("--profile"),
        std::ffi::OsString::from("harness-capability"),
    ];
    let client = HarnessClient::spawn(&argv, std::env::temp_dir().as_path())
        .await
        .expect("spawn dsh --profile harness-capability");
    let handshake = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        client.request("get_capabilities", json!({})),
    )
    .await
    .expect("handshake timed out")
    .expect("handshake failed");
    assert_eq!(handshake["runtime"], "deepseek-harness");
    assert_eq!(handshake["profile"], "harness-capability");
    client
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a locally built DSH checkout (env-gated)"]
async fn e2e_real_dsh_runtime_capabilities() {
    let client = spawn_runtime().await;

    // get_context: policy source of truth responds with sandbox fields.
    let context = client
        .request("get_context", json!({}))
        .await
        .expect("get_context");
    assert!(context["sandbox"]["mode"].is_string());
    assert!(context["sandbox"]["workspaceRoot"].is_string());

    // create_scope then scopeId existence check (B8 semantics: no behavior change).
    let scope = client
        .request("create_scope", json!({ "key": "e2e-scope" }))
        .await
        .expect("create_scope");
    assert_eq!(scope, json!({ "scopeId": "e2e-scope" }));
    let context_after = client
        .request("get_context", json!({}))
        .await
        .expect("get_context after create_scope");
    assert_eq!(context_after["scopes"], json!(["e2e-scope"]));

    // policy_check: denial is a result, never an error.
    let denied = client
        .request("policy_check", json!({ "operation": "write", "path": "/definitely/outside/file.txt" }))
        .await
        .expect("policy_check denied path");
    assert_eq!(denied["decision"], "denied");

    let allowed = client
        .request(
            "policy_check",
            json!({ "operation": "read", "path": "/anywhere/file.txt", "scopeId": "e2e-scope" }),
        )
        .await
        .expect("policy_check read");
    assert_eq!(allowed["decision"], "allowed");

    // Concurrent in-flight requests against the real runtime.
    let mut handles = Vec::new();
    for i in 0..4u64 {
        let client = Arc::clone(&client);
        handles.push(tokio::spawn(async move {
            client
                .request("get_context", json!({ "seq": i }))
                .await
                .expect("concurrent get_context")
        }));
    }
    for handle in handles {
        handle.await.expect("join");
    }

    // Unknown method maps to a remote error frame.
    match client.request("no_such", json!({})).await {
        Err(codex_harness_client::HarnessRequestError::Remote { message, .. }) => {
            assert!(message.contains("unknown"));
        }
        other => panic!("expected remote error, got {other:?}"),
    }

    client.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a locally built DSH checkout (env-gated)"]
async fn e2e_manager_lifecycle_launch_contract() {
    // The launch contract is profile-only (SPEC §10: not an arbitrary launcher).
    // SAFETY: single-threaded test setup, no concurrent env readers.
    unsafe { std::env::set_var("CODEX_HARNESS_DSH_BIN", dsh_bin()) };
    let argv = codex_harness_client::launch_argv("harness-capability").expect("launch argv");
    assert_eq!(argv[2], std::ffi::OsString::from("--profile"));
    assert_eq!(argv[3], std::ffi::OsString::from("harness-capability"));
    let manager = HarnessRuntimeManager::disabled();
    assert!(!manager.is_available());
}
