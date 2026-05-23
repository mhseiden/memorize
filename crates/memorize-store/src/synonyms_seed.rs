//! Initial synonym seed. Single-developer, code-heavy corpus — terms here are
//! ones I'd actually search for. Bidirectional pairs; each pair becomes two
//! rows at insert time.
//!
//! Add new pairs at runtime with `memorize syn add <a> <b>`; this list is only
//! consulted on first DB init (tracked in the `meta` table) so user deletions
//! aren't resurrected on later startups.

pub const DEFAULT_PAIRS: &[(&str, &str)] = &[
    ("k8s", "kubernetes"),
    ("kube", "kubernetes"),
    ("db", "database"),
    ("repo", "repository"),
    ("ci", "continuous integration"),
    ("cd", "continuous deployment"),
    ("pr", "pull request"),
    ("env", "environment"),
    ("api", "endpoint"),
    ("ui", "interface"),
    ("auth", "authentication"),
    ("authz", "authorization"),
    ("fe", "frontend"),
    ("be", "backend"),
    ("ts", "typescript"),
    ("js", "javascript"),
    ("rs", "rust"),
    ("py", "python"),
    ("config", "configuration"),
    ("docs", "documentation"),
    ("e2e", "end to end"),
    ("oom", "out of memory"),
    ("perf", "performance"),
    ("dev", "development"),
    ("prod", "production"),
    ("stg", "staging"),
];
