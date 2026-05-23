use regex::RegexSet;
use regex::Regex;

/// Patterns we redact from observation bodies before persistence. Conservative
/// — false negatives let secrets through, false positives are harmless
/// (replace with [REDACTED]). Each pattern is paired with its replacement
/// regex so we can do the actual substitution per-match.
const PATTERNS: &[&str] = &[
    // OpenAI-style: sk-... long-token
    r"sk-[A-Za-z0-9_-]{20,}",
    // Anthropic-style: sk-ant-...
    r"sk-ant-[A-Za-z0-9_-]{20,}",
    // GitHub PAT
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    // AWS access key id
    r"AKIA[0-9A-Z]{16}",
    // AWS secret-key-shaped 40-char base64
    r"(?i)aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}",
    // Bearer auth header value
    r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{20,}",
    // Generic API-key-looking assignments (key=..., api_key=...)
    r#"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*["']?[A-Za-z0-9._~+/=-]{16,}"#,
];

pub struct PrivacyFilter {
    set: RegexSet,
    individual: Vec<Regex>,
}

impl PrivacyFilter {
    pub fn new() -> Self {
        let set = RegexSet::new(PATTERNS).expect("privacy patterns must compile");
        let individual = PATTERNS
            .iter()
            .map(|p| Regex::new(p).expect("privacy patterns must compile"))
            .collect();
        Self { set, individual }
    }

    /// True if the input matches any secret pattern.
    pub fn has_secret(&self, body: &str) -> bool {
        self.set.is_match(body)
    }

    /// Redact every match in place. Cheap path: if `has_secret` is false we
    /// skip the full pass.
    pub fn redact(&self, body: &str) -> String {
        if !self.has_secret(body) {
            return body.to_string();
        }
        let mut out = body.to_string();
        for re in &self.individual {
            out = re.replace_all(&out, "[REDACTED]").into_owned();
        }
        out
    }
}

impl Default for PrivacyFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_key_redacted() {
        let p = PrivacyFilter::new();
        let s = "my key is sk-abcdef1234567890ABCDEF and that's it";
        assert!(p.has_secret(s));
        let r = p.redact(s);
        assert!(!r.contains("sk-abc"));
        assert!(r.contains("[REDACTED]"));
    }

    #[test]
    fn github_pat_redacted() {
        let p = PrivacyFilter::new();
        let s = "token=ghp_abcdef1234567890ABCDEFghij";
        let r = p.redact(s);
        assert!(!r.contains("ghp_abc"));
    }

    #[test]
    fn aws_access_key_redacted() {
        let p = PrivacyFilter::new();
        let s = "AKIAIOSFODNN7EXAMPLE";
        assert!(p.has_secret(s));
    }

    #[test]
    fn bearer_redacted() {
        let p = PrivacyFilter::new();
        let s = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.foo.bar";
        assert!(p.has_secret(s));
        let r = p.redact(s);
        assert!(!r.contains("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"));
    }

    #[test]
    fn ordinary_text_untouched() {
        let p = PrivacyFilter::new();
        let s = "Just a regular tool output with no secrets";
        assert!(!p.has_secret(s));
        assert_eq!(p.redact(s), s);
    }
}
