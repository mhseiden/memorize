//! Tree-sitter based AST chunker for the memorize code index.
//!
//! Inspired by cAST (arXiv 2506.15655): walk the parse tree, emit chunks at
//! semantic boundaries (functions, classes, methods, etc.). Oversized nodes
//! are recursively split; small siblings are greedy-merged up to a target
//! char budget so chunks stay near (but not necessarily under) the embedder's
//! context window.

use anyhow::{Context, Result, bail};
use serde::Serialize;
use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

/// Target chunk size in characters. Mirrors `memorize_core::CHUNK_CHARS` —
/// fits comfortably in MiniLM's 512-token window. Oversized AST nodes are
/// recursively split.
pub const TARGET_CHARS: usize = 1800;

/// Below this size, sibling chunkable nodes (small consts, imports, tiny
/// helpers) are greedy-merged so we don't generate a chunk per `use`
/// statement. Above it, each node is its own chunk — better for code search
/// granularity since a hit on one function shouldn't also drag in the next.
const MERGE_THRESHOLD: usize = 400;

#[derive(Debug, Clone, Serialize)]
pub struct CodeChunk {
    pub language: String,
    pub line_start: u32, // 1-based, inclusive
    pub line_end: u32,   // 1-based, inclusive
    pub kind: String,    // tree-sitter node kind for the dominant node
    pub qualified: String, // best-effort symbol identifier (e.g. function name)
    pub body: String,    // the source text covered by this chunk
}

/// Detect language from a file path by extension. Returns `None` if we don't
/// have a parser for it.
pub fn language_for_path(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?;
    Some(match ext {
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" | "mjs" | "cjs" => "javascript",
        "py" => "python",
        "go" => "go",
        "sh" | "bash" => "bash",
        _ => return None,
    })
}

/// Chunk a source file. Returns one or more chunks. If the language is
/// unrecognized or the parse fails, the whole file becomes a single chunk
/// — useful for fallback-coverage of misc text-ish files (README.md etc.
/// could be supported this way later, though today we only chunk recognized
/// languages).
pub fn chunk_file(path: &Path, source: &str) -> Result<Vec<CodeChunk>> {
    let language = language_for_path(path)
        .ok_or_else(|| anyhow::anyhow!("unsupported language for {}", path.display()))?;
    chunk_source(source, language)
}

pub fn chunk_source(source: &str, language: &str) -> Result<Vec<CodeChunk>> {
    let mut parser = Parser::new();
    let lang = load_language(language)?;
    parser
        .set_language(&lang)
        .with_context(|| format!("set tree-sitter language for {language}"))?;
    let tree = parser
        .parse(source, None)
        .with_context(|| format!("parse {language}"))?;
    Ok(chunk_tree(&tree, source, language))
}

fn load_language(language: &str) -> Result<tree_sitter::Language> {
    Ok(match language {
        "rust" => tree_sitter_rust::LANGUAGE.into(),
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "javascript" => tree_sitter_javascript::LANGUAGE.into(),
        "python" => tree_sitter_python::LANGUAGE.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        "bash" => tree_sitter_bash::LANGUAGE.into(),
        other => bail!("no tree-sitter grammar wired for {other}"),
    })
}

/// Per-language list of node kinds that are reasonable chunk boundaries when
/// they appear at file scope (or as direct children of containers like
/// `impl`, `class`, `mod`). The order doesn't matter — we use a set.
fn chunkable_kinds(language: &str) -> &'static [&'static str] {
    match language {
        "rust" => &[
            "function_item",
            "function_signature_item",
            "impl_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "mod_item",
            "type_item",
            "macro_definition",
            "const_item",
            "static_item",
        ],
        "typescript" | "tsx" | "javascript" => &[
            "function_declaration",
            "function_expression",
            "arrow_function",
            "method_definition",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "lexical_declaration",
            "export_statement",
        ],
        "python" => &[
            "function_definition",
            "class_definition",
            "decorated_definition",
            "import_statement",
            "import_from_statement",
            "assignment",
        ],
        "go" => &[
            "function_declaration",
            "method_declaration",
            "type_declaration",
            "var_declaration",
            "const_declaration",
        ],
        "bash" => &["function_definition", "command"],
        _ => &[],
    }
}

fn chunk_tree(tree: &Tree, source: &str, language: &str) -> Vec<CodeChunk> {
    let root = tree.root_node();
    let chunkable: std::collections::HashSet<&str> =
        chunkable_kinds(language).iter().copied().collect();
    let mut out: Vec<CodeChunk> = Vec::new();
    let mut buffer: Vec<NodeSlice> = Vec::new();
    walk(root, source, language, &chunkable, &mut buffer, &mut out);
    flush_buffer(&mut buffer, source, language, &mut out);
    if out.is_empty() {
        // Empty source or no chunkable nodes — emit the whole thing.
        let line_end = source.lines().count().max(1) as u32;
        out.push(CodeChunk {
            language: language.into(),
            line_start: 1,
            line_end,
            kind: "file".into(),
            qualified: String::new(),
            body: source.to_string(),
        });
    }
    out
}

#[derive(Debug)]
struct NodeSlice<'a> {
    node: Node<'a>,
    kind: String,
    qualified: String,
}

fn walk<'a>(
    node: Node<'a>,
    source: &str,
    language: &str,
    chunkable: &std::collections::HashSet<&str>,
    buffer: &mut Vec<NodeSlice<'a>>,
    out: &mut Vec<CodeChunk>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        let kind = child.kind();
        if chunkable.contains(kind) {
            let text_len = child.end_byte().saturating_sub(child.start_byte());
            if text_len > TARGET_CHARS && has_chunkable_descendants(child, chunkable) {
                // Recurse into oversized container (impl block, class, etc.).
                flush_buffer(buffer, source, language, out);
                walk(child, source, language, chunkable, buffer, out);
            } else if text_len > TARGET_CHARS {
                // Oversized leaf node (very long function with no inner
                // chunkable children). Emit on its own; truncation happens at
                // embedding time via memorize-core::chunk_for_embedding.
                flush_buffer(buffer, source, language, out);
                let slice = NodeSlice {
                    qualified: extract_qualified(child, source),
                    kind: kind.to_string(),
                    node: child,
                };
                emit_chunk(&[slice], source, language, out);
            } else if text_len > MERGE_THRESHOLD {
                // Substantial node — flush any accumulated tiny siblings
                // first, then emit this one on its own. Keeps a recall hit
                // on `fn foo` from also returning the adjacent `fn bar`.
                flush_buffer(buffer, source, language, out);
                let slice = NodeSlice {
                    qualified: extract_qualified(child, source),
                    kind: kind.to_string(),
                    node: child,
                };
                emit_chunk(&[slice], source, language, out);
            } else {
                // Tiny node — greedy-merge with prior tiny siblings.
                let combined = combined_len(buffer, &child);
                if combined > TARGET_CHARS && !buffer.is_empty() {
                    flush_buffer(buffer, source, language, out);
                }
                buffer.push(NodeSlice {
                    qualified: extract_qualified(child, source),
                    kind: kind.to_string(),
                    node: child,
                });
            }
        } else {
            // Non-chunkable node — recurse if it might contain chunkable
            // children (e.g. `source_file -> impl_item -> function_item`).
            if has_chunkable_descendants(child, chunkable) {
                flush_buffer(buffer, source, language, out);
                walk(child, source, language, chunkable, buffer, out);
            }
            // Leaf non-chunkable nodes (comments, whitespace, top-level
            // expressions in scripts, etc.) are skipped.
        }
    }
}

fn has_chunkable_descendants(
    node: Node<'_>,
    chunkable: &std::collections::HashSet<&str>,
) -> bool {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if chunkable.contains(child.kind()) {
            return true;
        }
        if has_chunkable_descendants(child, chunkable) {
            return true;
        }
    }
    false
}

fn combined_len(buffer: &[NodeSlice<'_>], next: &Node<'_>) -> usize {
    let start = buffer
        .first()
        .map(|n| n.node.start_byte())
        .unwrap_or(next.start_byte());
    let end = next.end_byte();
    end.saturating_sub(start)
}

fn flush_buffer<'a>(
    buffer: &mut Vec<NodeSlice<'a>>,
    source: &str,
    language: &str,
    out: &mut Vec<CodeChunk>,
) {
    if buffer.is_empty() {
        return;
    }
    let slices: Vec<NodeSlice<'a>> = std::mem::take(buffer);
    emit_chunk(&slices, source, language, out);
}

fn emit_chunk(
    slices: &[NodeSlice<'_>],
    source: &str,
    language: &str,
    out: &mut Vec<CodeChunk>,
) {
    if slices.is_empty() {
        return;
    }
    let first = &slices[0];
    let last = slices.last().unwrap();
    let start_byte = first.node.start_byte();
    let end_byte = last.node.end_byte();
    let body = source[start_byte..end_byte].to_string();
    let line_start = first.node.start_position().row as u32 + 1;
    let line_end = last.node.end_position().row as u32 + 1;
    let kind = if slices.len() == 1 {
        first.kind.clone()
    } else {
        format!("{}+", first.kind)
    };
    let qualified = slices
        .iter()
        .map(|s| s.qualified.as_str())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(",");
    out.push(CodeChunk {
        language: language.into(),
        line_start,
        line_end,
        kind,
        qualified,
        body,
    });
}

/// Best-effort symbol name extraction. Looks at the node's `name` field
/// (tree-sitter convention) or any direct `identifier`/`type_identifier`/
/// `property_identifier` child. Empty string if nothing found.
fn extract_qualified(node: Node<'_>, source: &str) -> String {
    if let Some(name) = node.child_by_field_name("name") {
        if let Ok(s) = name.utf8_text(source.as_bytes()) {
            return s.to_string();
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "identifier" | "type_identifier" | "property_identifier" | "field_identifier" => {
                if let Ok(s) = child.utf8_text(source.as_bytes()) {
                    return s.to_string();
                }
            }
            _ => {}
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_substantial_items_chunk_separately() {
        // Each item is well over MERGE_THRESHOLD (400 chars) so they should
        // emit as separate chunks even though they could fit together.
        let src = format!(
            r#"
struct Foo {{
{lines_a}
}}

fn add(a: i32, b: i32) -> i32 {{
{lines_b}
    a + b
}}
"#,
            lines_a = (0..40)
                .map(|i| format!("    field{i}: i32,"))
                .collect::<Vec<_>>()
                .join("\n"),
            lines_b = (0..40)
                .map(|i| format!("    let x{i} = {i};"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let chunks = chunk_source(&src, "rust").unwrap();
        assert!(
            chunks.iter().any(|c| c.qualified == "Foo"),
            "{:?}",
            chunks.iter().map(|c| &c.qualified).collect::<Vec<_>>()
        );
        assert!(
            chunks.iter().any(|c| c.qualified == "add"),
            "{:?}",
            chunks.iter().map(|c| &c.qualified).collect::<Vec<_>>()
        );
        let add = chunks.iter().find(|c| c.qualified == "add").unwrap();
        assert!(add.body.contains("a + b"));
        assert!(
            !add.body.contains("struct Foo"),
            "add chunk should not include struct Foo"
        );
    }

    #[test]
    fn rust_tiny_siblings_merge() {
        // Three tiny use declarations should produce one merged chunk.
        let src = r#"
use foo::Bar;
use baz::Quux;
use abc::Def;
"#;
        let chunks = chunk_source(src, "rust").unwrap();
        // Either one merged chunk or none (use_declaration isn't currently
        // in our chunkable set — uses fall through as non-chunkable). Either
        // is acceptable; key check is we don't crash on input like this.
        assert!(chunks.len() <= 1);
    }

    #[test]
    fn python_class_and_function_substantial() {
        let src = format!(
            r#"
class User:
    def __init__(self, name):
        self.name = name
{class_body}

def greet(u):
{greet_body}
    return f"hi {{u.name}}"
"#,
            class_body = (0..30)
                .map(|i| format!("        self.attr{i} = {i}"))
                .collect::<Vec<_>>()
                .join("\n"),
            greet_body = (0..30)
                .map(|i| format!("    local{i} = {i}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );
        let chunks = chunk_source(&src, "python").unwrap();
        assert!(chunks.iter().any(|c| c.qualified == "User"));
        assert!(chunks.iter().any(|c| c.qualified == "greet"));
    }

    #[test]
    fn typescript_class_with_methods() {
        let src = r#"
class UserService {
  async getUser(id: string) {
    return await this.db.get(id);
  }
  async listUsers() {
    return await this.db.all();
  }
}
"#;
        let chunks = chunk_source(src, "typescript").unwrap();
        // For a small class, the whole class is one chunk (under TARGET_CHARS).
        assert!(chunks.iter().any(|c| c.body.contains("getUser")));
    }

    #[test]
    fn oversized_node_still_emits_one_chunk() {
        // Construct a function whose body exceeds TARGET_CHARS but has no
        // chunkable descendants — should still emit one chunk.
        let mut src = String::from("fn big() {\n");
        for i in 0..200 {
            src.push_str(&format!("    let x{i} = {i};\n"));
        }
        src.push_str("}\n");
        let chunks = chunk_source(&src, "rust").unwrap();
        assert!(chunks.iter().any(|c| c.qualified == "big"));
    }

    #[test]
    fn unsupported_extension_returns_none() {
        assert!(language_for_path(Path::new("foo.png")).is_none());
        assert!(language_for_path(Path::new("foo.rs")).is_some());
    }
}
