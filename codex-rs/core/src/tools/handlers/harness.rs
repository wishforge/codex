use std::sync::Arc;

use crate::function_tool::FunctionCallError;
use crate::tools::context::FunctionToolOutput;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::handlers::parse_arguments;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExecutor;
use crate::tools::registry::ToolExposure;
use codex_harness_client::HarnessRequestError;
use codex_harness_client::HarnessRuntimeManager;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolExecutorFuture;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;

/// Fixed tool names, rendered exactly as `harness.<tool>` on the wire.
pub(crate) const HARNESS_GET_CONTEXT: &str = "harness.get_context";
pub(crate) const HARNESS_GET_CAPABILITIES: &str = "harness.get_capabilities";
pub(crate) const HARNESS_CREATE_SCOPE: &str = "harness.create_scope";
pub(crate) const HARNESS_POLICY_CHECK: &str = "harness.policy_check";

/// JSON-RPC method invoked on the DSH runtime for each tool.
const GET_CONTEXT_METHOD: &str = "get_context";
const GET_CAPABILITIES_METHOD: &str = "get_capabilities";
const CREATE_SCOPE_METHOD: &str = "create_scope";
const POLICY_CHECK_METHOD: &str = "policy_check";

/// One harness capability tool: a thin adapter from the function-tool calling
/// convention to one fixed JSON-RPC method on the session-owned DSH runtime.
///
/// The handler only *references* the runtime via [`HarnessRuntimeManager`];
/// it never spawns or shuts it down (SPEC §3.1: tool registration ≠ runtime
/// process lifecycle).
pub(crate) struct HarnessToolHandler {
    method: &'static str,
    tool_name: ToolName,
    spec: ToolSpec,
    manager: Arc<HarnessRuntimeManager>,
}

impl HarnessToolHandler {
    fn new(
        method: &'static str,
        tool_name: &'static str,
        description: &str,
        parameters: JsonSchema,
        manager: Arc<HarnessRuntimeManager>,
    ) -> Self {
        Self {
            method,
            tool_name: ToolName::plain(tool_name),
            spec: ToolSpec::Function(ResponsesApiTool {
                name: tool_name.to_string(),
                description: description.to_string(),
                strict: false,
                defer_loading: None,
                parameters,
                output_schema: None,
            }),
            manager,
        }
    }

    /// Build the four fixed harness handlers sharing one runtime manager.
    pub(crate) fn tool_set(manager: Arc<HarnessRuntimeManager>) -> Vec<Arc<Self>> {
        vec![
            Arc::new(Self::new(
                GET_CONTEXT_METHOD,
                HARNESS_GET_CONTEXT,
                "Return harness runtime context: runtime identity, profile, sandbox mode, \
                 workspace root, and known scope ids.",
                JsonSchema::object(BTreeMapJson::new(), None, Some(false.into())),
                Arc::clone(&manager),
            )),
            Arc::new(Self::new(
                GET_CAPABILITIES_METHOD,
                HARNESS_GET_CAPABILITIES,
                "List the harness runtime's capabilities (introspection only; never drives \
                 tool registration).",
                JsonSchema::object(BTreeMapJson::new(), None, Some(false.into())),
                Arc::clone(&manager),
            )),
            Arc::new(Self::new(
                CREATE_SCOPE_METHOD,
                HARNESS_CREATE_SCOPE,
                "Create a harness scope tagged with the given key. The scope lives until the \
                 harness runtime shuts down.",
                JsonSchema::object(
                    BTreeMapJson::from_iter([(
                        "key".to_string(),
                        JsonSchema::string(Some(
                            "Scope key to tag the new scope with.".to_string(),
                        )),
                    )]),
                    Some(vec!["key".to_string()]),
                    Some(false.into()),
                ),
                Arc::clone(&manager),
            )),
            Arc::new(Self::new(
                POLICY_CHECK_METHOD,
                HARNESS_POLICY_CHECK,
                "Check a filesystem operation against the harness sandbox policy. Denial is \
                 returned as a normal result, not an error. `scopeId` is an existence check \
                 only.",
                JsonSchema::object(
                    BTreeMapJson::from_iter([
                        (
                            "operation".to_string(),
                            JsonSchema::string_enum(
                                vec![json!("read"), json!("write")],
                                Some("Operation to check.".to_string()),
                            ),
                        ),
                        (
                            "path".to_string(),
                            JsonSchema::string(Some("Path to check.".to_string())),
                        ),
                        (
                            "scopeId".to_string(),
                            JsonSchema::string(Some(
                                "Optional scope id to check for existence.".to_string(),
                            )),
                        ),
                    ]),
                    Some(vec!["operation".to_string(), "path".to_string()]),
                    Some(false.into()),
                ),
                Arc::clone(&manager),
            )),
        ]
    }

    /// Validate model-supplied arguments and build the JSON-RPC params.
    fn build_params(&self, arguments: &str) -> Result<Value, FunctionCallError> {
        match self.method {
            CREATE_SCOPE_METHOD => {
                let params: CreateScopeParams = parse_arguments(arguments)?;
                Ok(json!({ "key": params.key }))
            }
            POLICY_CHECK_METHOD => {
                let params: PolicyCheckParams = parse_arguments(arguments)?;
                let mut params_object = json!({
                    "operation": params.operation,
                    "path": params.path,
                });
                // `scopeId` is optional in the wire contract (SPEC §7): omit the
                // key entirely when absent. A JSON `null` would be rejected by
                // the runtime's type check (`typeof scopeId !== 'string'`).
                if let Some(scope_id) = params.scope_id {
                    params_object["scopeId"] = Value::String(scope_id);
                }
                Ok(params_object)
            }
            // get_context / get_capabilities take an empty params object. Some
            // callers send an empty argument string instead of `{}`; normalize
            // it the way the MCP hook input does (`handlers/mcp.rs`).
            _ => {
                if arguments.trim().is_empty() {
                    return Ok(json!({}));
                }
                let _empty: EmptyParams = parse_arguments(arguments)?;
                Ok(json!({}))
            }
        }
    }
}

/// Dispatch one request against the runtime and map errors to model-facing
/// messages. Shared by all four handlers and unit-testable without a Session.
async fn invoke_harness(
    manager: &HarnessRuntimeManager,
    method: &str,
    params: Value,
) -> Result<Value, FunctionCallError> {
    manager.request(method, params).await.map_err(|error| {
        FunctionCallError::RespondToModel(match error {
            HarnessRequestError::Unavailable(reason) => {
                format!("harness runtime unavailable: {reason}")
            }
            HarnessRequestError::Remote { code, message } => {
                format!("harness runtime error {code}: {message}")
            }
            HarnessRequestError::Transport(message) => {
                format!("harness transport error: {message}")
            }
        })
    })
}

impl ToolExecutor<ToolInvocation> for HarnessToolHandler {
    fn tool_name(&self) -> ToolName {
        self.tool_name.clone()
    }

    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Direct
    }

    fn handle<'a>(&'a self, invocation: ToolInvocation) -> ToolExecutorFuture<'a>
    where
        ToolInvocation: 'a,
    {
        Box::pin(self.handle_call(invocation))
    }
}

impl HarnessToolHandler {
    async fn handle_call(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ToolInvocation { payload, .. } = invocation;
        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "harness tool handler received unsupported payload".to_string(),
                ));
            }
        };
        let params = self.build_params(&arguments)?;
        let result = invoke_harness(&self.manager, self.method, params).await?;
        let text = serde_json::to_string(&result).map_err(|error| {
            FunctionCallError::RespondToModel(format!(
                "failed to serialize harness tool result: {error}"
            ))
        })?;
        Ok(boxed_tool_output(FunctionToolOutput::from_content(
            vec![FunctionCallOutputContentItem::InputText { text }],
            /*success*/ Some(true),
        )))
    }
}

impl CoreToolRuntime for HarnessToolHandler {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmptyParams {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateScopeParams {
    key: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct PolicyCheckParams {
    operation: PolicyOperation,
    path: String,
    #[serde(default)]
    scope_id: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum PolicyOperation {
    Read,
    Write,
}

/// Alias keeping the schema construction calls within a reasonable width.
type BTreeMapJson = std::collections::BTreeMap<String, JsonSchema>;

#[cfg(test)]
mod tests {
    use super::*;
    use codex_harness_client::HarnessClient;
    use pretty_assertions::assert_eq;

    fn disabled_manager() -> Arc<HarnessRuntimeManager> {
        Arc::new(HarnessRuntimeManager::disabled())
    }

    /// Responder fixture: rewrites each echoed request frame into a JSON-RPC
    /// response whose `result` is the full request object. (`/bin/cat` cannot
    /// serve here: the client decodes frames with a top-level `method` as
    /// requests, never as responses. `fflush()` keeps the pipe line-buffered.)
    async fn echo_manager() -> Arc<HarnessRuntimeManager> {
        let script = r#"{ id=$0; sub(/^.*"id":/,"",id); sub(/,.*/,"",id); printf "{\"jsonrpc\":\"2.0\",\"id\":%s,\"result\":%s}\n", id, $0; fflush(); }"#;
        let client = HarnessClient::spawn(
            &[
                std::ffi::OsString::from("awk"),
                std::ffi::OsString::from(script),
            ],
            std::env::temp_dir().as_path(),
        )
        .await
        .expect("spawn echo fixture");
        Arc::new(HarnessRuntimeManager::Running(client))
    }

    /// One-shot JSON-RPC responder that replies with an error frame.
    async fn error_manager(code: i64, message: &str) -> Arc<HarnessRuntimeManager> {
        let script = format!(
            "read line; printf '%s\\n' '{{\"jsonrpc\":\"2.0\",\"id\":1,\"error\":{{\"code\":{code},\"message\":\"{message}\"}}}}'"
        );
        let client = HarnessClient::spawn(
            &[
                std::ffi::OsString::from("sh"),
                std::ffi::OsString::from("-c"),
                std::ffi::OsString::from(script),
            ],
            std::env::temp_dir().as_path(),
        )
        .await
        .expect("spawn error fixture");
        Arc::new(HarnessRuntimeManager::Running(client))
    }

    #[tokio::test]
    async fn disabled_manager_maps_to_unavailable_error() {
        let error = invoke_harness(&disabled_manager(), GET_CAPABILITIES_METHOD, json!({}))
            .await
            .expect_err("disabled manager must fail");
        match error {
            FunctionCallError::RespondToModel(message) => {
                assert_eq!(message, "harness runtime unavailable: harness is disabled");
            }
            other => panic!("expected RespondToModel, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn available_manager_returns_result_value() {
        let manager = echo_manager().await;
        let result = invoke_harness(&manager, GET_CAPABILITIES_METHOD, json!({}))
            .await
            .expect("echo fixture must answer");
        assert_eq!(result["method"], "get_capabilities");
        manager.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn remote_error_maps_to_model_message() {
        let manager = error_manager(-32602, "invalid params").await;
        let error = invoke_harness(&manager, CREATE_SCOPE_METHOD, json!({"key": "k"}))
            .await
            .expect_err("error frame must surface");
        match error {
            FunctionCallError::RespondToModel(message) => {
                assert_eq!(message, "harness runtime error -32602: invalid params");
            }
            other => panic!("expected RespondToModel, got {other:?}"),
        }
        manager.shutdown().await.expect("shutdown");
    }

    #[tokio::test]
    async fn parameter_validation_rejects_missing_and_unknown_fields() {
        let handlers = HarnessToolHandler::tool_set(disabled_manager());
        let by_name = |name: &str| {
            handlers
                .iter()
                .find(|handler| handler.tool_name.name == name)
                .expect("handler")
        };

        // create_scope requires `key`.
        let handler = by_name(HARNESS_CREATE_SCOPE);
        assert!(handler.build_params("{}").is_err());
        assert!(handler.build_params(r#"{"key":"k"}"#).is_ok());
        assert!(handler.build_params(r#"{"key":"k","extra":1}"#).is_err());

        // policy_check requires `operation` and `path`; scopeId is optional.
        let handler = by_name(HARNESS_POLICY_CHECK);
        assert!(handler.build_params("{}").is_err());
        assert!(handler.build_params(r#"{"operation":"read"}"#).is_err());
        assert!(
            handler
                .build_params(r#"{"operation":"delete","path":"/a"}"#)
                .is_err()
        );
        let params = handler
            .build_params(r#"{"operation":"write","path":"/a","scopeId":"s1"}"#)
            .expect("valid policy_check params");
        assert_eq!(params["operation"], "write");
        assert_eq!(params["scopeId"], "s1");
        let params = handler
            .build_params(r#"{"operation":"read","path":"/a"}"#)
            .expect("valid policy_check params without scopeId");
        // Optional `scopeId` must be omitted, not sent as JSON null: the runtime
        // rejects a non-string scopeId and would fail every scope-less call.
        assert!(
            params.get("scopeId").is_none(),
            "scopeId must be omitted when absent, got {params}"
        );

        // get_context / get_capabilities take no parameters.
        for name in [HARNESS_GET_CONTEXT, HARNESS_GET_CAPABILITIES] {
            let handler = by_name(name);
            assert!(handler.build_params("{}").is_ok());
            assert!(handler.build_params(r#"{"x":1}"#).is_err());
        }
    }

    #[test]
    fn specs_expose_fixed_dotted_names() {
        let handlers = HarnessToolHandler::tool_set(disabled_manager());
        let mut names: Vec<String> = handlers
            .iter()
            .map(|handler| handler.tool_name.name.clone())
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "harness.create_scope".to_string(),
                "harness.get_capabilities".to_string(),
                "harness.get_context".to_string(),
                "harness.policy_check".to_string(),
            ]
        );
        for handler in &handlers {
            let ToolSpec::Function(tool) = &handler.spec else {
                panic!("expected function spec");
            };
            assert_eq!(tool.name, handler.tool_name.name);
        }
    }
}
