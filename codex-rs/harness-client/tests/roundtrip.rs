//! Correlation, framing, and lifecycle tests against an awk responder fixture:
//! each echoed request frame is rewritten into a JSON-RPC response whose
//! `result` is the full request object, so response ids round-trip exactly
//! like a real JSON-RPC responder.

use std::ffi::OsString;
use std::sync::Arc;

use codex_harness_client::HarnessClient;
use codex_harness_client::HarnessRuntimeManager;
use serde_json::json;

/// `/bin/cat` cannot serve as a responder: it echoes the request frame
/// verbatim, and the client decodes any frame carrying a top-level `method` as
/// a Request, never as a Response (`exec-server-protocol/src/rpc.rs`). awk
/// rewrites each echoed request into a response frame; `fflush()` keeps it
/// line-buffered through the pipe (BSD sed would block until EOF).
fn responder_argv() -> Vec<OsString> {
    vec![
        OsString::from("awk"),
        OsString::from(
            r#"{ id=$0; sub(/^.*"id":/,"",id); sub(/,.*/,"",id); printf "{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":%s}\n", id, $0; fflush(); }"#,
        ),
    ]
}

#[tokio::test]
async fn request_correlates_response_by_id() {
    let client = HarnessClient::spawn(&responder_argv(), std::env::temp_dir().as_path())
        .await
        .expect("spawn echo fixture");
    let params = json!({"method": "get_capabilities"});
    let value = client
        .request("get_capabilities", params)
        .await
        .expect("response");
    // The awk fixture rewrites the echoed request into a response whose
    // `result` carries the full request object. Correlation is what this
    // proves: the reply for OUR id came back to OUR caller.
    assert_eq!(value["method"], "get_capabilities");
    client.shutdown().await.expect("graceful shutdown");
}

#[tokio::test]
async fn concurrent_requests_all_resolve() {
    let client = HarnessClient::spawn(&responder_argv(), std::env::temp_dir().as_path())
        .await
        .expect("spawn echo fixture");
    let mut handles = Vec::new();
    for i in 0..8u64 {
        let client = Arc::clone(&client);
        handles.push(tokio::spawn(async move {
            client
                .request("m", serde_json::json!({ "seq": i }))
                .await
                .expect("response")
        }));
    }
    for (i, handle) in handles.into_iter().enumerate() {
        let value = handle.await.expect("join");
        assert_eq!(value["params"]["seq"], json!(i as u64));
    }
    client.shutdown().await.expect("graceful shutdown");
}

#[tokio::test]
async fn shutdown_fails_pending_requests() {
    // `/bin/cat` echoes our frame verbatim; the client decodes that echo as a
    // Request (top-level `method`) and ignores it, so this request is never
    // answered — deterministic "in flight forever" — while cat still exits on
    // stdin EOF, so shutdown stays graceful.
    let argv = vec![OsString::from("/bin/cat")];
    let client = HarnessClient::spawn(&argv, std::env::temp_dir().as_path())
        .await
        .expect("spawn silent fixture");
    let client_for_request = Arc::clone(&client);
    let pending = tokio::spawn(async move {
        client_for_request
            .request("never_answered", serde_json::json!({}))
            .await
    });
    // Give the request time to register in the pending map.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    client.shutdown().await.expect("graceful shutdown");
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), pending)
        .await
        .expect("in-flight request must not hang after shutdown")
        .expect("join");
    assert!(matches!(
        outcome,
        Err(codex_harness_client::HarnessRequestError::Unavailable(_))
    ));

    // A request after shutdown fails fast, never hangs.
    let after = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        client.request("get_capabilities", serde_json::json!({})),
    )
    .await
    .expect("post-shutdown request must fail fast")
    .expect_err("runtime is shut down");
    assert!(matches!(
        after,
        codex_harness_client::HarnessRequestError::Unavailable(_)
    ));
}

#[tokio::test]
async fn manager_disabled_reports_unavailable() {
    let manager = HarnessRuntimeManager::disabled();
    assert!(!manager.is_available());
    assert!(manager.client().is_none());
    let error = manager
        .request("get_capabilities", serde_json::json!({}))
        .await
        .expect_err("disabled manager must fail");
    assert!(matches!(
        error,
        codex_harness_client::HarnessRequestError::Unavailable(_)
    ));
}
