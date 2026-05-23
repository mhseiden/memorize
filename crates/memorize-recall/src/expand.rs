use anyhow::Result;
use memorize_store::Store;

/// Tokenize a query for synonym lookup: lowercase, split on non-alphanumeric,
/// drop very short tokens. We don't try to be clever — DuckDB FTS does its own
/// Snowball stemming on the document side, so this just needs to extract the
/// rough word list to feed the synonyms table.
pub fn tokenize(query: &str) -> Vec<String> {
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(|t| t.to_lowercase())
        .collect()
}

/// Build the FTS query string from the original query plus synonym
/// expansions. DuckDB's `match_bm25` takes a space-separated bag of words and
/// treats them as implicit OR — exactly what we want for expansion.
pub fn build_fts_query(original_query: &str, store: &Store) -> Result<String> {
    let tokens = tokenize(original_query);
    if tokens.is_empty() {
        return Ok(original_query.to_string());
    }
    let expanded = store.expand_synonyms(&tokens)?;
    Ok(expanded.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_basic() {
        assert_eq!(
            tokenize("How does Kubernetes scheduling work?"),
            vec!["how", "does", "kubernetes", "scheduling", "work"]
        );
    }

    #[test]
    fn tokenize_drops_short() {
        let t = tokenize("a is the of");
        // "is", "the", "of" survive (>=2 chars); single-letter "a" drops.
        assert!(!t.contains(&"a".to_string()));
        assert!(t.contains(&"is".to_string()));
    }

    #[test]
    fn build_fts_query_includes_synonyms() {
        let store = Store::open_in_memory().unwrap();
        let q = build_fts_query("k8s scheduling", &store).unwrap();
        // Should contain at least the original tokens plus the seeded expansion.
        assert!(q.contains("k8s"));
        assert!(q.contains("kubernetes"));
        assert!(q.contains("scheduling"));
    }
}
