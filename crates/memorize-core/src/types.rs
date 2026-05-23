use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    UserPrompt,
    ToolUse,
    ToolFailure,
    SubagentStart,
    SubagentStop,
    TaskCompleted,
    SessionStart,
    SessionStop,
    Manual,
    Other,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::UserPrompt => "user_prompt",
            Kind::ToolUse => "tool_use",
            Kind::ToolFailure => "tool_failure",
            Kind::SubagentStart => "subagent_start",
            Kind::SubagentStop => "subagent_stop",
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
            "tool_use" => Kind::ToolUse,
            "tool_failure" => Kind::ToolFailure,
            "subagent_start" => Kind::SubagentStart,
            "subagent_stop" => Kind::SubagentStop,
            "task_completed" => Kind::TaskCompleted,
            "session_start" => Kind::SessionStart,
            "session_stop" => Kind::SessionStop,
            "manual" => Kind::Manual,
            _ => Kind::Other,
        }
    }
}

/// What hooks (and `memorize remember`) submit. Server allocates the id + ts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewObservation {
    pub session: String,
    pub kind: Kind,
    pub body: String,
    pub branch: Option<String>,
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
}

/// Tool-payload bodies can be very large (file contents, command output). We
/// truncate before persistence so a single Bash dump doesn't bloat the DB.
pub const MAX_BODY_BYTES: usize = 4096;

pub fn truncate_body(s: &str) -> String {
    if s.len() <= MAX_BODY_BYTES {
        return s.to_string();
    }
    // Cut on a char boundary to keep the string valid UTF-8.
    let mut cut = MAX_BODY_BYTES;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…[truncated]", &s[..cut])
}
