use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    UserPrompt,
    /// Final user-visible assistant message at end of a turn. Captured from
    /// the `last_assistant_message` field on the Stop hook payload.
    AssistantMessage,
    ToolUse,
    ToolFailure,
    SubagentStart,
    /// Subagent (Explore / Plan / etc.) final response — the research artifact.
    SubagentStop,
    SubagentMessage,
    TaskCompleted,
    SessionStart,
    SessionStop,
    Manual,
    #[default]
    Other,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::UserPrompt => "user_prompt",
            Kind::AssistantMessage => "assistant_message",
            Kind::ToolUse => "tool_use",
            Kind::ToolFailure => "tool_failure",
            Kind::SubagentStart => "subagent_start",
            Kind::SubagentStop => "subagent_stop",
            Kind::SubagentMessage => "subagent_message",
            Kind::TaskCompleted => "task_completed",
            Kind::SessionStart => "session_start",
            Kind::SessionStop => "session_stop",
            Kind::Manual => "manual",
            Kind::Other => "other",
        }
    }

    pub fn from_str(s: &str) -> Kind {
        match s {
            "user_prompt" => Kind::UserPrompt,
            "assistant_message" => Kind::AssistantMessage,
            "tool_use" => Kind::ToolUse,
            "tool_failure" => Kind::ToolFailure,
            "subagent_start" => Kind::SubagentStart,
            "subagent_stop" => Kind::SubagentStop,
            "subagent_message" => Kind::SubagentMessage,
            "task_completed" => Kind::TaskCompleted,
            "session_start" => Kind::SessionStart,
            "session_stop" => Kind::SessionStop,
            "manual" => Kind::Manual,
            _ => Kind::Other,
        }
    }
}

/// What hooks (and `memorize remember`) submit. Server allocates the id + ts.
/// `ref_*` fields are populated for tool-use obs (Read/Edit/Write/Bash) to
/// enable cheap filtering by file path without parsing the body string.
/// Prose obs (user_prompt / assistant_message / subagent_message / manual)
/// leave them None.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NewObservation {
    pub session: String,
    pub kind: Kind,
    pub body: String,
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub ref_path: Option<String>,
    #[serde(default)]
    pub ref_line_start: Option<i32>,
    #[serde(default)]
    pub ref_line_end: Option<i32>,
}

/// What recall returns. id + ts assigned at insert time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub id: i64,
    pub ts: i64,
    pub session: String,
    pub branch: Option<String>,
    pub kind: Kind,
    pub body: String,
    pub ref_path: Option<String>,
    pub ref_line_start: Option<i32>,
    pub ref_line_end: Option<i32>,
}

/// Tool-payload bodies can be very large (file contents, command output). We
/// truncate before persistence so a single Bash dump doesn't bloat the DB.
pub const MAX_BODY_BYTES: usize = 4096;

pub fn truncate_body(s: &str) -> String {
    if s.len() <= MAX_BODY_BYTES {
        return s.to_string();
    }
    let mut cut = MAX_BODY_BYTES;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…[truncated]", &s[..cut])
}

/// Char-window chunk size for embedding. MiniLM has a 512-token context;
/// at ~4 chars/token (English) 1800 chars leaves headroom for tokenization
/// variance. Bodies under this length produce exactly one chunk.
pub const CHUNK_CHARS: usize = 1800;

/// Split `body` into char-windowed chunks of at most CHUNK_CHARS bytes each.
/// Always returns at least one chunk (the empty body maps to a single empty
/// string — embedders accept this and produce a valid vector). Char-boundary
/// safe.
pub fn chunk_for_embedding(body: &str) -> Vec<&str> {
    if body.is_empty() {
        return vec![body];
    }
    if body.len() <= CHUNK_CHARS {
        return vec![body];
    }
    let mut chunks = Vec::with_capacity(body.len() / CHUNK_CHARS + 1);
    let mut start = 0;
    while start < body.len() {
        let mut end = (start + CHUNK_CHARS).min(body.len());
        while end > start && !body.is_char_boundary(end) {
            end -= 1;
        }
        chunks.push(&body[start..end]);
        if end == body.len() {
            break;
        }
        start = end;
    }
    chunks
}
