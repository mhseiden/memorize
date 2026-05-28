//! In-process BM25 indexes backed by tantivy. Replaces DuckDB's FTS extension,
//! which crashed under concurrent rebuild + write (the bg worker on one
//! connection raced the indexer's write tx on another).
//!
//! Two indexes live here: `obs` (single-field body) and `code` (body +
//! qualified + path_tokens, with stored path/language for filtering). Both
//! use `RamDirectory` — nothing on disk. The daemon rebuilds them at startup
//! by streaming `obs.body` / `code_chunks.body` from DuckDB; rebuild is
//! O(corpus) and one-shot, not periodic.
//!
//! ## Tokenization
//!
//! The DuckDB FTS path applied two regexes (camelCase split, then path-separator
//! split) to both the indexed text and the query string before handing them to
//! FTS. We preserve that contract with a custom `CodeTokenizer` that:
//!
//!   1. Walks the text byte-by-byte, splitting at non-alphanumerics, at lower→upper
//!      transitions (`fooBar` → `foo`, `Bar`), and at acronym-then-word boundaries
//!      (`IRMemo` → `IR`, `Memo`).
//!   2. Emits each sub-token to tantivy. Downstream filters lowercase and
//!      Snowball-stem.
//!
//! The obs index uses tantivy's stock `en_stem` (the default English analyzer)
//! since obs bodies are prose, not code identifiers.

use anyhow::{Context, Result};
use std::sync::Mutex;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, STORED, STRING, Schema, TEXT, Value};
use tantivy::tokenizer::{LowerCaser, Stemmer, TextAnalyzer, Token, TokenStream, Tokenizer};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

/// Heap budget for the tantivy `IndexWriter`. 50 MB is the recommended floor
/// from the tantivy docs and is plenty for our peak indexing burst (a few
/// dozen code chunks per file save).
const WRITER_HEAP_BYTES: usize = 50_000_000;

pub struct FtsIndex {
    obs: SingleFieldIndex,
    code: CodeIndex,
}

struct SingleFieldIndex {
    index: Index,
    reader: IndexReader,
    /// Tantivy `IndexWriter` is `Send` but the commit/delete API takes `&mut self`,
    /// so the surrounding `Store` would have to be `&mut` to call it. The writer
    /// lives behind its own mutex so the rest of the `Store` API can stay `&self`.
    writer: Mutex<IndexWriter>,
    id_field: Field,
    body_field: Field,
}

struct CodeIndex {
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    id_field: Field,
    path_field: Field,
    language_field: Field,
    body_field: Field,
    qualified_field: Field,
    path_tokens_field: Field,
}

#[derive(Debug, Clone)]
pub struct ObsHit {
    pub id: i64,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct CodeHit {
    pub id: i64,
    pub path: String,
    pub score: f64,
}

impl FtsIndex {
    pub fn new() -> Result<Self> {
        Ok(Self {
            obs: SingleFieldIndex::new_obs()?,
            code: CodeIndex::new()?,
        })
    }

    pub fn insert_obs(&self, id: i64, body: &str) -> Result<()> {
        self.obs.insert(id, body)
    }

    pub fn delete_obs(&self, id: i64) -> Result<()> {
        self.obs.delete(id)
    }

    pub fn search_obs(&self, query: &str, limit: usize) -> Result<Vec<ObsHit>> {
        self.obs.search(query, limit)
    }

    pub fn commit(&self) -> Result<()> {
        self.obs.commit()?;
        self.code.commit()?;
        Ok(())
    }

    pub fn insert_code(
        &self,
        id: i64,
        path: &str,
        language: &str,
        body: &str,
        qualified: &str,
        path_tokens: &str,
    ) -> Result<()> {
        self.code.insert(id, path, language, body, qualified, path_tokens)
    }

    pub fn delete_code(&self, id: i64) -> Result<()> {
        self.code.delete(id)
    }

    /// Drop every doc in the code index — used by `wipe_code_index`. The obs
    /// index isn't touched by that path.
    pub fn clear_code(&self) -> Result<()> {
        self.code.clear()
    }

    pub fn search_code(
        &self,
        query: &str,
        limit: usize,
        language: Option<&str>,
        path_prefix: Option<&str>,
    ) -> Result<Vec<CodeHit>> {
        self.code.search(query, limit, language, path_prefix)
    }
}

impl SingleFieldIndex {
    fn new_obs() -> Result<Self> {
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_i64_field("id", STORED | tantivy::schema::INDEXED);
        let body_field = schema_builder.add_text_field("body", TEXT);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let writer = index
            .writer(WRITER_HEAP_BYTES)
            .context("create obs index writer")?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("build obs index reader")?;
        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            id_field,
            body_field,
        })
    }

    fn insert(&self, id: i64, body: &str) -> Result<()> {
        let writer = self.writer.lock().unwrap();
        // Delete-then-add gives us upsert semantics. Same id replaces.
        writer.delete_term(Term::from_field_i64(self.id_field, id));
        let mut doc = TantivyDocument::default();
        doc.add_i64(self.id_field, id);
        doc.add_text(self.body_field, body);
        writer.add_document(doc).context("add obs doc")?;
        Ok(())
    }

    fn delete(&self, id: i64) -> Result<()> {
        let writer = self.writer.lock().unwrap();
        writer.delete_term(Term::from_field_i64(self.id_field, id));
        Ok(())
    }

    fn commit(&self) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();
        writer.commit().context("commit obs writer")?;
        drop(writer);
        self.reader.reload().context("reload obs reader")?;
        Ok(())
    }

    fn search(&self, query: &str, limit: usize) -> Result<Vec<ObsHit>> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(&self.index, vec![self.body_field]);
        // Lenient: tolerate stray operators and unknown fields. Errors are
        // dropped — the recall pipeline owns query construction and we'd
        // rather return *some* hits than fail the request.
        let (q, _errs) = parser.parse_query_lenient(query);
        let top = searcher.search(&q, &TopDocs::with_limit(limit).order_by_score())?;
        let mut out = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            let id = doc
                .get_first(self.id_field)
                .and_then(|v| v.as_i64())
                .context("obs hit missing id")?;
            out.push(ObsHit {
                id,
                score: score as f64,
            });
        }
        Ok(out)
    }
}

impl CodeIndex {
    fn new() -> Result<Self> {
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_i64_field("id", STORED | tantivy::schema::INDEXED);
        let path_field = schema_builder.add_text_field("path", STRING | STORED);
        let language_field = schema_builder.add_text_field("language", STRING | STORED);
        // The three searchable fields all run through our custom "code"
        // analyzer (camelCase + path-separator split, lowercase, snowball).
        let text_opts = tantivy::schema::TextOptions::default().set_indexing_options(
            tantivy::schema::TextFieldIndexing::default()
                .set_tokenizer("code")
                .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
        );
        let body_field = schema_builder.add_text_field("body", text_opts.clone());
        let qualified_field = schema_builder.add_text_field("qualified", text_opts.clone());
        let path_tokens_field = schema_builder.add_text_field("path_tokens", text_opts);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema);
        let analyzer = TextAnalyzer::builder(CodeTokenizer::default())
            .filter(LowerCaser)
            .filter(Stemmer::new(tantivy::tokenizer::Language::English))
            .build();
        index.tokenizers().register("code", analyzer);

        let writer = index
            .writer(WRITER_HEAP_BYTES)
            .context("create code index writer")?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .context("build code index reader")?;
        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            id_field,
            path_field,
            language_field,
            body_field,
            qualified_field,
            path_tokens_field,
        })
    }

    fn insert(
        &self,
        id: i64,
        path: &str,
        language: &str,
        body: &str,
        qualified: &str,
        path_tokens: &str,
    ) -> Result<()> {
        let writer = self.writer.lock().unwrap();
        writer.delete_term(Term::from_field_i64(self.id_field, id));
        let mut doc = TantivyDocument::default();
        doc.add_i64(self.id_field, id);
        doc.add_text(self.path_field, path);
        doc.add_text(self.language_field, language);
        doc.add_text(self.body_field, body);
        doc.add_text(self.qualified_field, qualified);
        doc.add_text(self.path_tokens_field, path_tokens);
        writer.add_document(doc).context("add code doc")?;
        Ok(())
    }

    fn delete(&self, id: i64) -> Result<()> {
        let writer = self.writer.lock().unwrap();
        writer.delete_term(Term::from_field_i64(self.id_field, id));
        Ok(())
    }

    fn clear(&self) -> Result<()> {
        let writer = self.writer.lock().unwrap();
        writer.delete_all_documents().context("clear code index")?;
        Ok(())
    }

    fn commit(&self) -> Result<()> {
        let mut writer = self.writer.lock().unwrap();
        writer.commit().context("commit code writer")?;
        drop(writer);
        self.reader.reload().context("reload code reader")?;
        Ok(())
    }

    fn search(
        &self,
        query: &str,
        limit: usize,
        language: Option<&str>,
        path_prefix: Option<&str>,
    ) -> Result<Vec<CodeHit>> {
        if query.trim().is_empty() {
            return Ok(vec![]);
        }
        let searcher = self.reader.searcher();
        let parser = QueryParser::for_index(
            &self.index,
            vec![self.body_field, self.qualified_field, self.path_tokens_field],
        );
        let (q, _errs) = parser.parse_query_lenient(query);
        // Over-fetch when filters are active so the post-filter still has
        // enough candidates to fill `limit`. 5× is enough at our scale; the
        // language/path-prefix filters are rarely highly selective in
        // practice.
        let fetch = if language.is_some() || path_prefix.is_some() {
            (limit * 5).max(limit)
        } else {
            limit
        };
        let top = searcher.search(&q, &TopDocs::with_limit(fetch).order_by_score())?;
        let mut out = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            let id = doc
                .get_first(self.id_field)
                .and_then(|v| v.as_i64())
                .context("code hit missing id")?;
            let path = doc
                .get_first(self.path_field)
                .and_then(|v| v.as_str())
                .context("code hit missing path")?
                .to_string();
            let lang = doc
                .get_first(self.language_field)
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if let Some(want_lang) = language
                && lang != want_lang
            {
                continue;
            }
            if let Some(prefix) = path_prefix
                && !path.starts_with(prefix)
            {
                continue;
            }
            out.push(CodeHit {
                id,
                path,
                score: score as f64,
            });
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }
}

/// Tokenizer that mirrors the SQL-side camelCase + path-separator split.
///
/// Boundaries (in order of precedence):
///   - any non-alphanumeric byte: split, drop the separator.
///   - lower→upper transition: split before the uppercase byte (`fooBar` →
///     `foo` | `Bar`).
///   - upper-run followed by upper-lower (`IRMemo` → `IR` | `Memo`): split
///     before the trailing capital that starts a new word.
///
/// Pure ASCII path. Non-ASCII bytes fall into the "non-alphanumeric" bucket
/// and act as separators — fine for our corpus (paths + source code).
#[derive(Clone, Default)]
struct CodeTokenizer {
    token: Token,
}

pub struct CodeTokenStream<'a> {
    text: &'a str,
    bytes: &'a [u8],
    cursor: usize,
    token: &'a mut Token,
    position: usize,
}

impl Tokenizer for CodeTokenizer {
    type TokenStream<'a> = CodeTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> CodeTokenStream<'a> {
        self.token.reset();
        CodeTokenStream {
            text,
            bytes: text.as_bytes(),
            cursor: 0,
            token: &mut self.token,
            position: 0,
        }
    }
}

impl<'a> TokenStream for CodeTokenStream<'a> {
    fn advance(&mut self) -> bool {
        let n = self.bytes.len();
        // Skip separators (non-alphanumeric ASCII, or any non-ASCII byte).
        while self.cursor < n && !is_alnum(self.bytes[self.cursor]) {
            self.cursor += 1;
        }
        if self.cursor >= n {
            return false;
        }
        let start = self.cursor;
        let first = self.bytes[start];
        self.cursor += 1;
        if first.is_ascii_uppercase() {
            // Consume more uppercase. If we hit a lowercase, decide whether
            // the *previous* uppercase started a new word (acronym→word
            // boundary like `IR` | `Memo` inside `IRMemo`).
            while self.cursor < n && self.bytes[self.cursor].is_ascii_uppercase() {
                self.cursor += 1;
            }
            // We're sitting either at end, at a separator, at a digit, or
            // at a lowercase letter. The acronym boundary only applies in the
            // last case AND only when we consumed more than one uppercase
            // letter — otherwise it's a normal mixedCase run we'll handle
            // by continuing through lowercases.
            if self.cursor < n
                && self.bytes[self.cursor].is_ascii_lowercase()
                && self.cursor - start >= 2
            {
                // Back off one so the trailing capital starts the next token.
                self.cursor -= 1;
            } else {
                // Continue consuming lowercases/digits as part of this token.
                while self.cursor < n && is_alnum_lower_or_digit(self.bytes[self.cursor]) {
                    self.cursor += 1;
                }
            }
        } else {
            // Started lowercase or digit: consume run of lowercase + digits,
            // stop at uppercase (camelCase boundary) or separator.
            while self.cursor < n && is_alnum_lower_or_digit(self.bytes[self.cursor]) {
                self.cursor += 1;
            }
        }
        let end = self.cursor;
        // The byte range `[start, end)` is ASCII-only (we only advanced over
        // ASCII alphanumerics), so it's a valid UTF-8 slice of `self.text`.
        let term = &self.text[start..end];
        self.token.text.clear();
        self.token.text.push_str(term);
        self.token.offset_from = start;
        self.token.offset_to = end;
        self.token.position = self.position;
        self.position = self.position.wrapping_add(1);
        true
    }

    fn token(&self) -> &Token {
        self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        self.token
    }
}

#[inline]
fn is_alnum(b: u8) -> bool {
    b.is_ascii_alphanumeric()
}

#[inline]
fn is_alnum_lower_or_digit(b: u8) -> bool {
    b.is_ascii_lowercase() || b.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(s: &str) -> Vec<String> {
        let mut tk = CodeTokenizer::default();
        let mut stream = tk.token_stream(s);
        let mut out = Vec::new();
        while stream.advance() {
            out.push(stream.token().text.clone());
        }
        out
    }

    #[test]
    fn camel_case_split() {
        assert_eq!(tokens("snapshotMemoGraph"), vec!["snapshot", "Memo", "Graph"]);
    }

    #[test]
    fn path_separator_split() {
        assert_eq!(
            tokens("packages/foo-bar/irMemo/memo.ts"),
            vec!["packages", "foo", "bar", "ir", "Memo", "memo", "ts"]
        );
    }

    #[test]
    fn acronym_then_word() {
        assert_eq!(tokens("IRMemo"), vec!["IR", "Memo"]);
        assert_eq!(tokens("HTTPSConnection"), vec!["HTTPS", "Connection"]);
    }

    #[test]
    fn digits_attach_to_run() {
        assert_eq!(tokens("foo42bar"), vec!["foo42bar"]);
        assert_eq!(tokens("v2Engine"), vec!["v2", "Engine"]);
    }

    #[test]
    fn empty() {
        assert_eq!(tokens(""), Vec::<String>::new());
        assert_eq!(tokens("///---"), Vec::<String>::new());
    }

    #[test]
    fn obs_index_basic_roundtrip() {
        let idx = FtsIndex::new().unwrap();
        idx.insert_obs(1, "learned about kubernetes pod scheduling").unwrap();
        idx.insert_obs(2, "rust borrow checker notes").unwrap();
        idx.commit().unwrap();
        let hits = idx.search_obs("kubernetes", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn obs_upsert_replaces() {
        let idx = FtsIndex::new().unwrap();
        idx.insert_obs(1, "first body about cats").unwrap();
        idx.commit().unwrap();
        idx.insert_obs(1, "second body about dogs").unwrap();
        idx.commit().unwrap();
        let hits = idx.search_obs("cats", 10).unwrap();
        assert!(hits.is_empty(), "old body should not be searchable");
        let hits = idx.search_obs("dogs", 10).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn obs_delete() {
        let idx = FtsIndex::new().unwrap();
        idx.insert_obs(1, "transient note").unwrap();
        idx.commit().unwrap();
        idx.delete_obs(1).unwrap();
        idx.commit().unwrap();
        assert!(idx.search_obs("transient", 10).unwrap().is_empty());
    }

    #[test]
    fn code_index_camel_case_query() {
        let idx = FtsIndex::new().unwrap();
        idx.insert_code(
            1,
            "src/snapshot_memo_graph.rs",
            "rust",
            "fn snapshotMemoGraph() {}",
            "snapshotMemoGraph",
            "src snapshot memo graph rs",
        )
        .unwrap();
        idx.commit().unwrap();
        // Query expressed as camelCase — should match because both sides
        // tokenize through the same analyzer.
        let hits = idx.search_code("snapshotMemoGraph", 10, None, None).unwrap();
        assert_eq!(hits.len(), 1);
        // Single-word query that matches one piece of the split should also hit.
        let hits = idx.search_code("snapshot", 10, None, None).unwrap();
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn code_index_path_prefix_filter() {
        let idx = FtsIndex::new().unwrap();
        idx.insert_code(1, "src/a.rs", "rust", "alpha", "alpha", "src a rs").unwrap();
        idx.insert_code(2, "tests/b.rs", "rust", "alpha", "alpha", "tests b rs").unwrap();
        idx.commit().unwrap();
        let hits = idx.search_code("alpha", 10, None, Some("src/")).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);
    }

    #[test]
    fn code_index_language_filter() {
        let idx = FtsIndex::new().unwrap();
        idx.insert_code(1, "src/a.rs", "rust", "alpha", "alpha", "src a rs").unwrap();
        idx.insert_code(2, "src/b.ts", "typescript", "alpha", "alpha", "src b ts").unwrap();
        idx.commit().unwrap();
        let hits = idx.search_code("alpha", 10, Some("rust"), None).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 1);
    }
}
