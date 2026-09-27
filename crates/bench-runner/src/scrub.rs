//! #134 — scrubbing for engine-controlled text on its way into a SEALED artifact.
//!
//! ## Why this exists
//!
//! Before #134 the worker's stderr tail never reached a persisted record: it was forwarded to a
//! local terminal (never on official runs) and then discarded with the transport. #134 opened that
//! channel deliberately — an infra reject has to carry the engine's own last words or it cannot be
//! diagnosed — and opening it created a NEW way for engine-controlled bytes to land in
//! `results.json` / `score.*.json`.
//!
//! [`redact_worker_stderr_line`](crate::redact_worker_stderr_line) is NOT sufficient for that. It
//! is a keyword filter (`expected` / `actual`) ported from Swift, aimed at ONE threat: a submitted
//! model leaking golden tokens. It does nothing about the shape of secret that a crashing worker
//! actually prints — absolute paths naming the private golden and the operator's home directory,
//! `user@host` strings, and `KEY=VALUE` environment dumps. A line with none of its trigger
//! keywords passes through byte-for-byte.
//!
//! ## Secret-tier rule
//!
//! Program rule: endpoint hostnames, R2/bucket URLs, golden paths and M5 credentials live in
//! `.env` / ssh-config ONLY, and are referenced elsewhere by alias — never reproduced in repos,
//! issues, PRs or artifacts. A sealed `results.json` is an artifact and travels with the run, so
//! this module fails SAFE: anything path-shaped, host-shaped, URL-shaped or credential-shaped is
//! replaced with a sentinel, and only the non-sensitive remainder (exit status, error class,
//! prose, and a path's BASENAME) survives — which is the part that carries the diagnosis.
//!
//! Over-redaction is the intended failure direction: losing a directory name costs a little
//! context, sealing a secret costs a rotation.

/// Replaces the directory part of an absolute path; the basename is kept because it is what names
/// the failure (`config.json`, `model.safetensors`).
const PATH_SENTINEL: &str = "<path>";
/// Replaces any token carrying a URI scheme (`https://…`, `s3://…`, `file://…`).
const URL_SENTINEL: &str = "<url>";
/// Replaces any `user@host`-shaped token (also covers bare email addresses).
const USERHOST_SENTINEL: &str = "<user@host>";
/// Replaces the VALUE of a `KEY=VALUE` / `KEY: VALUE` token whose key looks credential-bearing.
const SECRET_SENTINEL: &str = "<redacted>";
/// Replaces a bare hostname / FQDN (optionally keeping a `:port`).
const HOST_SENTINEL: &str = "<host>";

/// Hosts kept READABLE. A loopback name carries no deployment identity and is a genuinely useful
/// diagnostic ("the worker dialled itself"), so it is deliberately not redacted. Every other
/// dotted name is treated as potentially secret-tier. `127.0.0.1` needs no entry — an all-numeric
/// final label is already excluded by [`is_fqdn`].
const READABLE_HOSTS: &[&str] = &["localhost", "localhost.localdomain"];

/// Final labels that mark a dotted token as a FILENAME rather than a hostname.
///
/// `sample-001.json` and `api.example.internal` are the same shape — dotted labels with an
/// alphabetic final label — so the two are separated by this allowlist. An unknown extension is
/// therefore treated as a host and redacted: over-redaction is the safe direction, and the
/// basename is only diagnostic sugar whereas an endpoint is secret-tier.
///
/// ACCEPTED over-redaction, and a DELIBERATE non-fix: a missing space after a full stop makes
/// prose the same shape too, so `init failed.See log` seals as `init <host> log`. An allowlist
/// cannot fix that (`See` is not an extension), and the obvious sentence-boundary guard —
/// "a capitalised final label is prose, not a TLD" — was REJECTED: it would let `API.EXAMPLE.INTERNAL`
/// through, and an uppercase endpoint is exactly the secret-tier string this module exists to
/// stop. A TLD allowlist fails for the same reason in reverse, since `.internal` is not a real
/// TLD. Losing one prose word costs a little context; leaking the endpoint costs a rotation.
const FILE_EXTENSIONS: &[&str] = &[
    "json",
    "safetensors",
    "txt",
    "log",
    "py",
    "rs",
    "swift",
    "toml",
    "yaml",
    "yml",
    "md",
    "sh",
    "so",
    "dylib",
    "bin",
    "npz",
    "gguf",
    "csv",
    "tsv",
    "sha256",
    "lock",
    "cfg",
    "ini",
    "plist",
    "sb",
    "h",
    "c",
    "cpp",
    "hpp",
    "o",
    "a",
    "gz",
    "zip",
    "tar",
    "png",
    "jpg",
    "svg",
    "html",
    "xml",
    // MLX / Metal write targets. These sharpen the hypothesis-(b) discriminator: a Seatbelt
    // denial is diagnosed from WHICH file the engine could not write, and without these entries
    // `default.metallib` was indistinguishable from an FQDN and sealed as `<host>` — destroying
    // the one token that names the finding.
    //
    // ACCEPTED TRADE, stated because every extension added here WIDENS what a hostname can hide
    // behind: a dotted token ending in one of these labels is no longer sealed, so a name like
    // `gw.example.metal` or `node1.cluster.air` would now emit verbatim where it previously
    // sealed as `<host>`. That is accepted on a checked premise, not on hope — no program
    // endpoint ends in any of these labels, and the suffixes the program's real endpoints DO use
    // (`.internal`, `.fail`, `.org`, `.com`) are absent from this list and still seal. The
    // tripwire test `mlx_metal_write_targets_stay_readable_but_hosts_do_not` pins that premise,
    // so if a future endpoint ever takes one of these suffixes the decision gets revisited
    // deliberately rather than discovered in an artifact.
    "metallib",
    "metal",
    "air",
    "npy",
    "jsonl",
    "framework",
    "mlpackage",
    "gputrace",
];

/// Substrings (matched case-insensitively against the KEY of a `KEY=VALUE` / `KEY: VALUE` token)
/// that mark the value as secret- or identity-bearing. Deliberately broad — `KEY` alone catches
/// `AWS_SECRET_ACCESS_KEY`, `HOST` catches `host=`, `HOME`/`USER`/`LOGNAME` catch the operator
/// identity an env dump carries, and `PATH`/`DIR` catch private locations. A false positive costs
/// one diagnostic value; a false negative costs a rotation.
///
/// ACCEPTED over-redaction: model-config lines share this vocabulary, so
/// `num_key_value_heads=8` seals as `num_key_value_heads=<redacted>` (`KEY`) — and, since a fired
/// key now takes the REST OF THE LINE (see [`credential_slot`]), the config values after it go too.
/// That is the fail-safe direction and is not treated as a defect — the KEY still names where the
/// line was cut, and no scoring path reads these strings. The alternative is worse: the reason
/// this rule had to widen is that redacting only the first slot left the SECOND word
/// (`Authorization: Bearer sk-live-…`) sealed verbatim.
const CREDENTIAL_KEY_MARKERS: &[&str] = &[
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "TOKEN",
    "CREDENTIAL",
    "AUTH",
    "BEARER",
    "SIGNATURE",
    "SESSION",
    "COOKIE",
    "PRIVATE",
    "ENDPOINT",
    "URL",
    "URI",
    "HOST",
    "KEY",
    "HOME",
    "USER",
    "LOGNAME",
    "ACCOUNT",
    "EMAIL",
    "PATH",
    "DIR",
];

/// Cap on a reason string sealed into an artifact. The retained stderr tail is bounded at 64 KiB,
/// but a sealed record should not carry 64 KiB of engine-controlled text PER REJECT — the full
/// (scrubbed) tail goes to the log, the record gets this much.
pub const SEALED_REASON_BYTE_LIMIT: usize = 2048;

/// Marker left in place of the bytes a cap removed.
const CLIPPED_MARKER: &str = "…[clipped]…";

/// Scrub one line of engine-controlled text: strip raw control bytes, then replace every
/// path-, URL-, host- and credential-shaped token with a sentinel.
///
/// Idempotent, so it is safe to apply at the source AND again at the seal boundary.
pub fn scrub_engine_text(line: &str) -> String {
    let controlled = strip_control_bytes(line);
    let mut out = String::with_capacity(controlled.len());
    let mut rest = controlled.as_str();
    while !rest.is_empty() {
        // Copy the run of separators verbatim (space / tab survive `strip_control_bytes`).
        let token_start = rest
            .find(|c: char| c != ' ' && c != '\t')
            .unwrap_or(rest.len());
        out.push_str(&rest[..token_start]);
        rest = &rest[token_start..];
        if rest.is_empty() {
            break;
        }
        let token_end = rest.find([' ', '\t']).unwrap_or(rest.len());
        let token = &rest[..token_end];
        let remainder = &rest[token_end..];

        // A credential KEY seals its value wherever the delimiter puts it — see
        // [`credential_slot`]. Once the key has fired, the REST OF THE LINE goes: with
        // `Authorization:` the secret is the SECOND word (`Bearer sk-live-…`), so redacting only
        // the value slot, or only the next token, still leaks it.
        if let Some((prefix, bridge)) = credential_slot(token, remainder) {
            out.push_str(prefix);
            out.push_str(bridge);
            out.push_str(SECRET_SENTINEL);
            break;
        }

        out.push_str(&scrub_token(token));
        rest = remainder;
    }
    out
}

/// Does `token` open a CREDENTIAL VALUE SLOT — a credential-bearing key followed by a delimiter?
///
/// If so, returns `(prefix, bridge)`: the leading slice of `token` to keep verbatim (key plus its
/// delimiter) and the text that bridges it to the sentinel. The caller emits
/// `prefix + bridge + <redacted>` and DISCARDS THE REST OF THE LINE.
///
/// The delimiter can put the value in three different places, and before this the per-token rules
/// only ever saw one of them — a key that fired still leaked, because the redaction landed on the
/// wrong slot. All four forms are the same leak:
///
/// | form | token | value actually sits |
/// |---|---|---|
/// | header | `Authorization:` | next token onward (`Bearer sk-live-…`) |
/// | attached-colon | `X-Auth-Token:Bearer` | inside the token, continuing after it |
/// | equals | `Authorization=Bearer` | inside the token, continuing after it |
/// | bare-colon | `Authorization` + `:` | after a delimiter token of its own |
///
/// Key matching is case-insensitive ([`is_credential_key`] upper-cases), so `authorization=bearer`
/// is the same finding as `Authorization=Bearer`.
fn credential_slot<'a>(token: &'a str, remainder: &str) -> Option<(&'a str, &'static str)> {
    // A URI carries `:` and `=` of its own and is sealed WHOLE by `scrub_core`; splitting it here
    // would keep its scheme and eat the line instead.
    if token.contains("://") {
        return None;
    }
    let (prefix, bridge, value) = match token.find(['=', ':']) {
        // Header form: the delimiter ends the token, so the value is everything after it.
        Some(i) if i + 1 == token.len() => (token, " ", next_token(remainder)),
        // Attached form: the value slot opens inside the token.
        Some(i) => (&token[..=i], "", &token[i + 1..]),
        // Bare-colon form: the delimiter is a token of its own, so NEITHER half is
        // credential-shaped alone. Normalise it back onto the key — `Authorization : Bearer x`
        // seals exactly like `Authorization: Bearer x`.
        None if next_token(remainder) == ":" => (
            token,
            ": ",
            next_token(
                remainder
                    .trim_start_matches([' ', '\t'])
                    .trim_start_matches(':'),
            ),
        ),
        _ => return None,
    };

    let key = strip_quotes(prefix.trim_end_matches(['=', ':']));
    if key.is_empty()
        // `localhost:8080` — the one host policy keeps readable, and it contains the `HOST` marker.
        || is_readable_host(key)
        // A `=`/`:` inside prose or a wrapper-laden JSON blob is not an assignment at all; those
        // stay on the `scrub_token` path, which splits the wrappers first.
        || !looks_like_assignment_key(key)
        || !is_credential_key(key)
    {
        return None;
    }
    // Nothing in the slot (`AWS_SECRET_ACCESS_KEY:` at end of line) — no value to leak.
    //
    // Already the sentinel: IDEMPOTENCE. The seal boundary re-scrubs a string whose lines were
    // scrubbed individually and then joined, and eating the remainder on that second pass would
    // destroy every later line's diagnosis.
    if value.is_empty() || value == SECRET_SENTINEL {
        return None;
    }
    // `auth.example.com:443` is a host:PORT wearing a credential-ish name. Leave it to
    // `scrub_host_like`, which seals the host and KEEPS the diagnostic port, rather than eating
    // the rest of the line behind it.
    if is_fqdn(key) && value.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some((prefix, bridge))
}

/// The next whitespace-delimited token in `rest`, or `""` if there is none.
fn next_token(rest: &str) -> &str {
    let after_ws = rest.trim_start_matches([' ', '\t']);
    &after_ws[..after_ws.find([' ', '\t']).unwrap_or(after_ws.len())]
}

/// Strip any surrounding ASCII quotes, so `"Authorization"` tests the same as `Authorization`.
fn strip_quotes(s: &str) -> &str {
    s.trim_matches(['"', '\''])
}

/// Escape anything that is not printable text. A worker that dies inside a binary read can emit
/// raw ESC / BEL / NUL, which would otherwise ride into a JSON record and into any terminal that
/// later prints it (ESC sequences can rewrite a reviewer's screen). Tab survives; the tail is
/// already newline-split, so any remaining `\n`/`\r` is escaped too.
fn strip_control_bytes(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    for ch in line.chars() {
        match ch {
            '\t' => out.push('\t'),
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Scrub one whitespace-delimited token by splitting it on WRAPPER punctuation and scrubbing each
/// piece, so a path embedded mid-token still matches (`open("/Users/x/g.json"),` →
/// `open("<path>/g.json"),`). Delimiters are re-emitted verbatim.
fn scrub_token(token: &str) -> String {
    let mut out = String::with_capacity(token.len());
    let mut piece_start = 0;
    for (i, c) in token.char_indices() {
        if !is_wrapper(c) {
            continue;
        }
        if piece_start < i {
            out.push_str(&scrub_core(&token[piece_start..i]));
        }
        out.push(c);
        piece_start = i + c.len_utf8();
    }
    if piece_start < token.len() {
        out.push_str(&scrub_core(&token[piece_start..]));
    }
    out
}

/// Punctuation that can WRAP a path/host without being part of it.
///
/// `/`, `~`, `.`, `_`, `-` are excluded because they start or appear inside the very things being
/// matched. `:` is excluded because it is part of a URI scheme (`https://`), which must be seen
/// whole. `<`/`>` are excluded because they delimit this module's own sentinels — treating them as
/// wrappers would re-split an already-scrubbed string and break idempotence.
fn is_wrapper(c: char) -> bool {
    matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | '`')
}

fn scrub_core(core: &str) -> String {
    if core.is_empty() {
        return String::new();
    }
    // Quotes are handled HERE rather than as wrappers, so a quoted JSON pair
    // (`"AWS_SECRET_ACCESS_KEY":"wJal…"`) stays ONE core and its key/value relationship survives
    // to the rules below. Splitting on quotes first would hide it.
    let unquoted = core.trim_start_matches(['"', '\'']);
    let lead_len = core.len() - unquoted.len();
    let inner = unquoted.trim_end_matches(['"', '\'']);
    let trail_len = unquoted.len() - inner.len();
    if (lead_len > 0 || trail_len > 0) && !inner.is_empty() {
        return format!(
            "{}{}{}",
            &core[..lead_len],
            scrub_core(inner),
            &unquoted[inner.len()..]
        );
    }

    // Order matters. A URL can contain `@`, `:` and `/`, so it is decided first; host:port is
    // decided BEFORE the `:` credential rule, or `api.example.com:443` would be read as a
    // benign `KEY:VALUE` and kept whole.
    if core.contains("://") {
        return URL_SENTINEL.to_string();
    }
    if let Some(eq) = core.find('=') {
        let (key, value) = core.split_at(eq);
        let value = &value[1..];
        if !key.is_empty() && looks_like_assignment_key(strip_quotes(key)) {
            if is_credential_key(strip_quotes(key)) {
                return format!("{key}={SECRET_SENTINEL}");
            }
            // Benign key, but the VALUE can still be a path, a host or a URL.
            return format!("{key}={}", scrub_core(value));
        }
    }
    if let Some(at) = core.find('@') {
        if at > 0 && at + 1 < core.len() {
            return USERHOST_SENTINEL.to_string();
        }
    }
    if is_path_like(core) {
        return abbreviate_path(core);
    }
    if let Some(host) = scrub_host_like(core) {
        return host;
    }
    // COLON-form credential with no space (`AWS_SECRET_ACCESS_KEY:wJal…`, and the inner half of a
    // JSON pair). Only acted on when the key is credential-bearing: a benign `foo:bar` is left
    // alone so ordinary prose (`note:`, `errno:`) is not mangled.
    if let Some(colon) = core.find(':') {
        let (key, value) = core.split_at(colon);
        let value = &value[1..];
        // A READABLE host is explicitly non-secret, and `localhost` contains the `HOST` marker —
        // without this guard `localhost:8080` would seal as `localhost:<redacted>`, redacting the
        // one host the policy says to keep.
        if !key.is_empty()
            && !value.is_empty()
            && !is_readable_host(strip_quotes(key))
            && looks_like_assignment_key(strip_quotes(key))
            && is_credential_key(strip_quotes(key))
        {
            return format!("{key}:{SECRET_SENTINEL}");
        }
    }
    core.to_string()
}

/// Is this token a filesystem path? Absolute, home-relative (`~/`, `~user/`) and explicitly
/// relative (`./`, `../`) forms all count, as does any token with three or more components —
/// `Users/operator/x` names an operator just as plainly as `/Users/operator/x`.
///
/// A bare two-component token (`read/write`, `and/or`) is NOT treated as a path, so ordinary prose
/// survives.
fn is_path_like(core: &str) -> bool {
    if !core.contains('/') {
        return false;
    }
    core.starts_with('/')
        || core.starts_with('~')
        || core.starts_with("./")
        || core.starts_with("../")
        || core.split('/').filter(|p| !p.is_empty()).count() >= 3
}

/// Replace a bare hostname / FQDN, keeping any `:port` (the port is diagnostic and not secret).
fn scrub_host_like(core: &str) -> Option<String> {
    if let Some(colon) = core.rfind(':') {
        let (host, port) = (&core[..colon], &core[colon + 1..]);
        if !port.is_empty() && port.chars().all(|c| c.is_ascii_digit()) && is_fqdn(host) {
            return Some(format!("{HOST_SENTINEL}:{port}"));
        }
    }
    is_fqdn(core).then(|| HOST_SENTINEL.to_string())
}

/// Is this one of the hosts policy keeps readable ([`READABLE_HOSTS`])?
fn is_readable_host(s: &str) -> bool {
    READABLE_HOSTS.contains(&s.to_ascii_lowercase().as_str())
}

/// Dotted-label heuristic for "this is a hostname, not a word or a filename".
fn is_fqdn(s: &str) -> bool {
    if s.is_empty() || is_readable_host(s) {
        return false;
    }
    if !s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
    {
        return false;
    }
    let labels: Vec<&str> = s.split('.').collect();
    if labels.len() < 2 || labels.iter().any(|l| l.is_empty()) {
        return false;
    }
    let last = labels[labels.len() - 1].to_ascii_lowercase();
    // An all-numeric final label is a dotted-quad (IP literal) — kept readable, like loopback:
    // it names no deployment and is useful in a connect failure.
    if last.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    last.chars().all(|c| c.is_ascii_alphabetic())
        && (2..=24).contains(&last.len())
        && !FILE_EXTENSIONS.contains(&last.as_str())
}

/// A key is assignment-shaped when it reads like an identifier — otherwise a `=` inside prose or
/// JSON is not an assignment at all.
fn looks_like_assignment_key(key: &str) -> bool {
    key.chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
        && key.chars().any(|c| c.is_ascii_alphabetic())
}

fn is_credential_key(key: &str) -> bool {
    let upper = key.to_ascii_uppercase();
    CREDENTIAL_KEY_MARKERS
        .iter()
        .any(|marker| upper.contains(marker))
}

/// Keep only the final component of an absolute path — the part that names the failure — and
/// replace every directory above it, which is where operator identity and private locations live.
///
/// A SHALLOW path (`/Users/operator`, `/home/operator`) is replaced WHOLE: its last component is not a
/// filename, it is the account name, so keeping it would defeat the point.
///
/// Navigation components (`.`, `..`) and a leading `~user` are dropped before the depth is
/// counted, so `../../Users/operator/x` is measured — and abbreviated — exactly like the absolute
/// path it resolves to.
///
/// The surviving basename is itself run back through [`scrub_core`]: a basename is still
/// engine-controlled text, and a file literally named `AWS_SECRET_ACCESS_KEY=wJal…` would
/// otherwise be emitted verbatim by the very function meant to be removing secrets.
fn abbreviate_path(path: &str) -> String {
    let parts: Vec<&str> = path
        .split('/')
        .filter(|p| !p.is_empty() && *p != "." && *p != ".." && !p.starts_with('~'))
        .collect();
    match parts.last() {
        Some(base) if parts.len() > 2 => format!("{PATH_SENTINEL}/{}", scrub_core(base)),
        _ => PATH_SENTINEL.to_string(),
    }
}

/// Scrub `reason` and cap it at [`SEALED_REASON_BYTE_LIMIT`] for a persisted record.
///
/// The ONE entry point every seal site calls, so a new sink cannot pick up an unscrubbed variant
/// by accident. Both halves matter: scrubbing keeps secrets out of the artifact, the cap keeps a
/// single engine-controlled reject from carrying kilobytes of engine text into it.
pub fn scrub_reason_for_seal(reason: &str) -> String {
    clip_to_bytes(&scrub_engine_text(reason), SEALED_REASON_BYTE_LIMIT)
}

/// The SEALED stand-in for engine-controlled text: `engine-text-sha256=<hex> engine-text-bytes=<n>`
/// of the SCRUBBED text, and none of the text itself.
///
/// The engine sees the hidden prompts and the sealed record is returned to the participant, so any
/// engine-chosen bytes that reached the record would be a channel for reading a hidden prompt back
/// (fail on purpose with the prompt in the reason). The digest lets an operator match a record to
/// the full text in the box-local log without the record carrying it.
pub fn seal_engine_text(text: &str) -> String {
    let scrubbed = scrub_engine_text(text);
    format!(
        "engine-text-sha256={} engine-text-bytes={}",
        bench_core::hash::sha256_hex(scrubbed.as_bytes()),
        scrubbed.len()
    )
}

/// Clip `text` to at most `cap` BYTES, keeping a head and a tail around a marker, never splitting
/// a UTF-8 character. Both ends are kept because the head names the failure and the tail is where
/// a worker's last words are.
pub(crate) fn clip_to_bytes(text: &str, cap: usize) -> String {
    if text.len() <= cap {
        return text.to_string();
    }
    if cap <= CLIPPED_MARKER.len() {
        return text[..floor_boundary(text, cap)].to_string();
    }
    let room = cap - CLIPPED_MARKER.len();
    let head_len = floor_boundary(text, room / 2);
    let tail_start = ceil_boundary(text, text.len() - (room - head_len));
    format!(
        "{}{CLIPPED_MARKER}{}",
        &text[..head_len],
        &text[tail_start..]
    )
}

/// Largest char boundary `<= i` (`str::floor_char_boundary` is still unstable).
fn floor_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Smallest char boundary `>= i`.
fn ceil_boundary(s: &str, mut i: usize) -> usize {
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A1 — the shapes the reviewer's probe sealed VERBATIM before this existed. None of them
    /// contains the `expected`/`actual` keywords the Swift-ported filter looks for, which is
    /// exactly why that filter did not stop them.
    #[test]
    fn secret_shaped_stderr_without_trigger_keywords_is_scrubbed() {
        let scrubbed = scrub_engine_text(
            "load failed /Users/operator/pool-goldens/sample-001.json AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI \
             host=api.example.internal user=operator@ranked-box HOME=/Users/operator",
        );
        // Absolute paths lose their directories, keep their basename.
        assert!(!scrubbed.contains("/Users/operator"), "{scrubbed}");
        assert!(scrubbed.contains("<path>/sample-001.json"), "{scrubbed}");
        // Credential values never survive.
        assert!(!scrubbed.contains("wJalrXUtnFEMI"), "{scrubbed}");
        assert!(
            scrubbed.contains("AWS_SECRET_ACCESS_KEY=<redacted>"),
            "{scrubbed}"
        );
        // Endpoint hostnames and user@host identities never survive.
        assert!(!scrubbed.contains("api.example.internal"), "{scrubbed}");
        assert!(!scrubbed.contains("operator@ranked-box"), "{scrubbed}");
        // The diagnosis itself is preserved.
        assert!(scrubbed.starts_with("load failed "), "{scrubbed}");
    }

    #[test]
    fn urls_and_schemes_are_replaced_whole() {
        let scrubbed = scrub_engine_text("PUT s3://bucket-name/key failed via https://u:p@h/x");
        assert!(!scrubbed.contains("bucket-name"), "{scrubbed}");
        assert!(!scrubbed.contains("u:p@h"), "{scrubbed}");
        assert_eq!(scrubbed, "PUT <url> failed via <url>");
    }

    /// Quoted / parenthesised / trailing-comma paths must still match.
    #[test]
    fn paths_wrapped_in_punctuation_are_still_scrubbed() {
        let scrubbed = scrub_engine_text("open(\"/Users/x/secret/weights.safetensors\"), errno=2");
        assert!(!scrubbed.contains("/Users/x/secret"), "{scrubbed}");
        assert!(scrubbed.contains("weights.safetensors"), "{scrubbed}");
        assert!(scrubbed.contains("errno=2"), "benign kv lost: {scrubbed}");
    }

    /// B3 — raw control bytes must not ride into a JSON record or a reviewer's terminal.
    #[test]
    fn control_bytes_are_escaped() {
        let scrubbed = scrub_engine_text("boom\u{1b}[31mRED\u{7}\u{0}end");
        assert!(!scrubbed.contains('\u{1b}'), "ESC survived: {scrubbed:?}");
        assert!(!scrubbed.contains('\u{7}'), "BEL survived: {scrubbed:?}");
        assert!(!scrubbed.contains('\u{0}'), "NUL survived: {scrubbed:?}");
        assert!(scrubbed.contains("\\x1b"), "not escaped: {scrubbed:?}");
        assert!(scrubbed.contains("end"), "content lost: {scrubbed:?}");
        // Tab is legitimate diagnostic layout and survives.
        assert_eq!(scrub_engine_text("a\tb"), "a\tb");
    }

    /// Ordinary diagnostic prose — the reason this channel was opened — must survive intact.
    #[test]
    fn diagnostic_prose_survives_unchanged() {
        for line in [
            "worker exited with status 3",
            "fatal: could not open weights",
            "CRITICAL: metal device init failed",
            "token-validation-failed",
            "mlxfast-swift: runtime worker parent exited; shutting down to release model memory",
        ] {
            assert_eq!(scrub_engine_text(line), line, "prose was mangled: {line}");
        }
    }

    /// Re-attack 1 (peer, verbatim strings) — a BARE hostname / FQDN and a `host:port`, the modal
    /// shape a dying worker prints an endpoint in. Neither is a `KEY=VALUE`, so the marker list
    /// never saw them and both sealed unchanged.
    #[test]
    fn bare_fqdn_and_host_port_are_replaced() {
        let a = scrub_engine_text("Failed to connect to api.example.internal port 443");
        assert!(!a.contains("api.example.internal"), "{a}");
        assert_eq!(a, "Failed to connect to <host> port 443");

        let b = scrub_engine_text("dial host-dev.example.fail:443");
        assert!(!b.contains("host-dev.example.fail"), "{b}");
        // The port is kept: it is diagnostic and names no deployment.
        assert_eq!(b, "dial <host>:443");
    }

    /// Loopback stays readable (documented choice) and a filename is not mistaken for a host —
    /// the two are the same dotted shape, separated only by the extension allowlist.
    #[test]
    fn loopback_and_filenames_are_not_treated_as_hosts() {
        assert_eq!(
            scrub_engine_text("bound localhost:8080"),
            "bound localhost:8080"
        );
        assert_eq!(
            scrub_engine_text("dial 127.0.0.1:9000"),
            "dial 127.0.0.1:9000"
        );
        assert_eq!(
            scrub_engine_text("open sample-001.json failed"),
            "open sample-001.json failed"
        );
        assert_eq!(
            scrub_engine_text("mmap weights.safetensors"),
            "mmap weights.safetensors"
        );
    }

    /// Re-attack 2 (peer, verbatim string) — RELATIVE and `~user` paths carried the operator name
    /// through whole, because only `/`- and `~/`-prefixed tokens reached `abbreviate_path`.
    #[test]
    fn relative_and_tilde_user_paths_are_abbreviated() {
        let a = scrub_engine_text("open ../../Users/operator/x failed");
        assert!(!a.contains("operator"), "operator name survived: {a}");
        assert_eq!(a, "open <path>/x failed");

        let b = scrub_engine_text("stat ~operator/secrets.txt");
        assert!(!b.contains("operator"), "operator name survived: {b}");

        let c = scrub_engine_text("read Users/operator/pool-goldens/sample-001.json");
        assert!(!c.contains("operator"), "operator name survived: {c}");
        assert!(c.contains("<path>/sample-001.json"), "diagnosis lost: {c}");

        // Two-component prose is NOT a path and must survive.
        assert_eq!(scrub_engine_text("mode read/write"), "mode read/write");
    }

    /// Re-attack 3 (peer, verbatim strings) — COLON-form credentials. The key and its value are
    /// different whitespace tokens, so no per-token rule could ever pair them; the JSON form is
    /// the same leak wearing quotes.
    #[test]
    fn colon_form_and_json_form_credentials_are_redacted() {
        let a = scrub_engine_text("AWS_SECRET_ACCESS_KEY: wJalrXUtnFEMIK7MDENGbPxRfiCY");
        assert!(!a.contains("wJalrXUtnFEMIK7MDENGbPxRfiCY"), "{a}");
        assert!(a.starts_with("AWS_SECRET_ACCESS_KEY:"), "{a}");

        // The secret is the SECOND word here, so redacting one token would still leak it.
        let b = scrub_engine_text("Authorization: Bearer sk-live-abc123def456");
        assert!(!b.contains("sk-live-abc123def456"), "{b}");
        assert!(!b.contains("Bearer sk-live"), "{b}");

        let c = scrub_engine_text("AWS_SECRET_ACCESS_KEY:wJalrXUtnFEMIK7MDENGbPxRfiCY");
        assert!(!c.contains("wJalrXUtnFEMIK7MDENGbPxRfiCY"), "{c}");

        // JSON env dump — also closes "env dumped as JSON" as a channel.
        let d = scrub_engine_text(r#"{"AWS_SECRET_ACCESS_KEY":"wJalrXUtnFEMIK7MDENGbPxRfiCY"}"#);
        assert!(!d.contains("wJalrXUtnFEMIK7MDENGbPxRfiCY"), "{d}");
        assert!(d.contains("AWS_SECRET_ACCESS_KEY"), "key label lost: {d}");
    }

    /// Re-attack 5 (reviewer, verbatim strings) — the VALUE-SLOT family. `is_credential_key` fired
    /// on every one of these, and every one still leaked: the redaction landed on the wrong slot.
    /// `Authorization: Bearer …` (the ONE form that was tested) sealed, which is precisely why the
    /// hole was invisible — the fix has to key on the KEY, not on the delimiter that follows it.
    #[test]
    fn a_fired_credential_key_seals_the_value_slot_in_every_delimiter_form() {
        const SECRET: &str = "sk-live-abc123def456";
        for line in [
            // Attached colon — value in the SAME token, secret in the next.
            "Authorization:Bearer sk-live-abc123def456",
            // Equals form.
            "Authorization=Bearer sk-live-abc123def456",
            // Bare-colon token: neither half is credential-shaped on its own.
            "Authorization : Bearer sk-live-abc123def456",
            // Sibling headers — the marker list already covered the keys, nothing acted on them.
            "Proxy-Authorization:Digest sk-live-abc123def456",
            "X-Auth-Token:Bearer sk-live-abc123def456",
            // Case variants: header keys are case-insensitive on the wire.
            "authorization: bearer sk-live-abc123def456",
            "AUTHORIZATION=BEARER sk-live-abc123def456",
        ] {
            let scrubbed = scrub_engine_text(line);
            assert!(
                !scrubbed.contains(SECRET),
                "secret survived: {line} -> {scrubbed}"
            );
            assert!(
                !scrubbed.to_ascii_lowercase().contains("bearer")
                    && !scrubbed.to_ascii_lowercase().contains("digest"),
                "auth scheme survived, so the slot was not consumed: {line} -> {scrubbed}"
            );
            assert!(
                scrubbed.contains(SECRET_SENTINEL),
                "nothing was sealed: {line} -> {scrubbed}"
            );
            // The KEY must survive: it names what was cut, and it is not itself secret.
            let key = line.split([':', '=', ' ']).next().unwrap();
            assert!(
                scrubbed.starts_with(key),
                "key label lost: {line} -> {scrubbed}"
            );
            // Every variant must also be idempotent — the family redacts to end-of-line, which is
            // the rule that nearly broke re-sealing before.
            assert_eq!(
                scrub_engine_text(&scrubbed),
                scrubbed,
                "not idempotent: {line}"
            );
        }
    }

    /// The controls for the family above: shapes that must NOT change behaviour. A rule that
    /// redacts to end-of-line is easy to make too eager, and these are the things it would eat.
    #[test]
    fn value_slot_rule_leaves_the_controls_alone() {
        // Already sealed before the fix — must still seal, and identically.
        let a = scrub_engine_text("Authorization: Bearer sk-live-abc123def456");
        assert_eq!(a, "Authorization: <redacted>", "{a}");

        let b = scrub_engine_text("token:sk-live-abc123def456");
        assert!(!b.contains("sk-live-abc123def456"), "{b}");
        assert_eq!(b, "token:<redacted>", "{b}");

        // A credential key with an EMPTY slot has nothing to leak and must not be rewritten.
        assert_eq!(
            scrub_engine_text("AWS_SECRET_ACCESS_KEY:"),
            "AWS_SECRET_ACCESS_KEY:"
        );

        // ACCEPTED over-redaction, pinned so it stays a known limit: a model-config key matches
        // `KEY`, and the widened rule now takes the rest of the line with it.
        assert_eq!(
            scrub_engine_text("num_key_value_heads=8 hidden_size=4096"),
            "num_key_value_heads=<redacted>"
        );

        // host:PORT must not be mistaken for KEY:VALUE — the port is diagnostic and is kept, and
        // the text behind it must survive.
        assert_eq!(
            scrub_engine_text("bound localhost:8080 ok"),
            "bound localhost:8080 ok"
        );
        assert_eq!(
            scrub_engine_text("dial host-dev.example.fail:443 refused"),
            "dial <host>:443 refused"
        );

        // A bare `:` after a NON-credential word is ordinary prose and must be left exactly alone.
        assert_eq!(scrub_engine_text("status : 3"), "status : 3");
    }

    /// #134/(b) — the MLX/Metal write discriminator. A Seatbelt denial is diagnosed from WHICH
    /// file the engine could not write, and `default.metallib` is the same dotted shape as an
    /// FQDN, so before the extension additions it sealed as `<host>` and took the finding with it.
    #[test]
    fn mlx_metal_write_targets_stay_readable_but_hosts_do_not() {
        let a =
            scrub_engine_text("cannot write /x/Library/Caches/com.apple.metal/default.metallib");
        assert!(
            a.contains("default.metallib"),
            "the filename that names the finding was sealed: {a}"
        );
        assert!(!a.contains("/x/Library"), "directories leaked: {a}");
        assert_eq!(a, "cannot write <path>/default.metallib");

        for name in [
            "kernel.metal",
            "shader.air",
            "tokens.npy",
            "pool.jsonl",
            "Metal.framework",
            "model.mlpackage",
            "capture.gputrace",
        ] {
            assert_eq!(
                scrub_engine_text(&format!("write {name} failed")),
                format!("write {name} failed"),
                "MLX/Metal write target sealed as a host: {name}"
            );
        }

        // The discriminator must not have been bought by weakening the host rule.
        assert_eq!(
            scrub_engine_text("Failed to connect to api.example.internal port 443"),
            "Failed to connect to <host> port 443"
        );

        // TRIPWIRE for the accepted trade (see FILE_EXTENSIONS): the suffixes the program's real
        // endpoints use must NOT be in the allowlist. If an endpoint ever takes one of the
        // extension labels, this is the row that has to be argued with first.
        //
        // REPRESENTATIVE HOSTS, deliberately — no real endpoint appears in this file. The premise
        // under test is SUFFIX-based (`.internal`/`.fail`/`.org` are absent from FILE_EXTENSIONS,
        // `.metal` is present), so a placeholder carrying the same final label proves exactly the
        // same property. Secret-tier rule: endpoints live in `.env`/ssh-config, and a test literal
        // is a repo literal like any other.
        for endpoint in [
            "api.example.internal",
            "host-dev.example.fail",
            "api.example.org",
            "gateway.example.com",
        ] {
            assert_eq!(
                scrub_engine_text(&format!("dial {endpoint}")),
                "dial <host>",
                "a program endpoint suffix leaked through the extension allowlist: {endpoint}"
            );
        }
        // And the cost side of the same trade, pinned honestly: a host that DOES end in one of the
        // new labels now survives. No such endpoint exists today; this records what we bought.
        assert_eq!(
            scrub_engine_text("dial gw.example.metal"),
            "dial gw.example.metal",
            "if this ever needs to seal, the extension trade has to be revisited"
        );
    }

    /// ACCEPTED RESIDUAL, pinned deliberately (see [`FILE_EXTENSIONS`]): a missing space after a
    /// full stop is the same dotted shape as an FQDN. The extension allowlist cannot fix this —
    /// `See` is not an extension — and the capitalisation-based sentence guard was REJECTED
    /// because it would let an uppercase endpoint through. Over-redaction is the ruled direction.
    #[test]
    fn a_missing_space_after_a_full_stop_over_redacts_by_design() {
        let scrubbed = scrub_engine_text("metal device init failed.See log");
        assert_eq!(
            scrubbed, "metal device init <host> log",
            "if this changes, the sentence-boundary decision was revisited — update FILE_EXTENSIONS' \
             doc comment and say why an uppercase FQDN is still safe"
        );
        // The thing that decision protects.
        assert_eq!(
            scrub_engine_text("connect API.EXAMPLE.INTERNAL failed"),
            "connect <host> failed"
        );
    }

    /// Re-attack 4 (peer) — `abbreviate_path` emitted the basename RAW, so a path whose BASENAME
    /// is a credential assignment sealed the secret verbatim out of the very function meant to be
    /// removing them.
    #[test]
    fn a_credential_shaped_basename_is_scrubbed_not_emitted_raw() {
        let a =
            scrub_engine_text("open /tmp/dump/AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCY");
        assert!(
            !a.contains("wJalrXUtnFEMIK7MDENGbPxRfiCY"),
            "credential basename emitted raw: {a}"
        );
        assert!(a.contains("AWS_SECRET_ACCESS_KEY=<redacted>"), "{a}");

        // Same hole one level down: a basename that is itself an endpoint.
        let b = scrub_engine_text("open /var/run/api.example.internal");
        assert!(!b.contains("api.example.internal"), "{b}");
    }

    /// ACCEPTED RESIDUAL, recorded so it is a known limit rather than a surprise: a secret that is
    /// base64- or percent-ENCODED has no `KEY=`, no dots and no `/`-shape, so nothing here matches
    /// it and it seals verbatim. The 2 KiB cap is NOT a mitigation — every realistic key fits well
    /// inside it. Closing this would need entropy heuristics, which would eat ordinary diagnostics
    /// (hashes, digests, token ids) and is out of scope for #134.
    #[test]
    fn encoded_secrets_are_a_documented_residual_not_a_claim() {
        let encoded = "d0phbHJYVXRuRkVNSUs3TURFTkdiUHhSZmlDWQ==";
        let scrubbed = scrub_engine_text(&format!("blob {encoded}"));
        assert!(
            scrubbed.contains(encoded),
            "if this ever starts passing, the residual is closed — update the docs"
        );
    }

    /// ACCEPTED RESIDUAL — the shapes a fired credential key still does NOT seal, recorded so the
    /// limit is known rather than a surprise. NONE is a regression: every one behaves identically
    /// on the pre-#134 code, so the value-slot fix strictly improved this surface without closing
    /// it. Tracked as #140.
    ///
    /// THREE DISTINCT CAUSES, which is why this is a follow-up and not a wider delimiter rule —
    /// each needs a different part of the token walk changed:
    ///
    /// 1. WRAPPER SPLIT. [`scrub_token`] splits a token on wrapper punctuation and scrubs each
    ///    piece through [`scrub_core`], so a wrapper-laden token never reaches the token-level
    ///    [`credential_slot`] rule. `scrub_core` can only redact the value slot inside its own
    ///    piece; the secret in the next whitespace token survives.
    /// 2. DETACHED-COLON SUFFIX. The bare-colon branch of [`credential_slot`] requires the next
    ///    token to be EXACTLY `":"`. `Authorization :Bearer …` puts the scheme in the same token
    ///    as the colon, so neither the in-token nor the bare-colon branch matches and the line is
    ///    left completely untouched — the worst of the three.
    /// 3. PATH ABSORPTION. [`is_path_like`] claims the token first, and [`abbreviate_path`]
    ///    replaces a shallow path WHOLE — so the credential key disappears into `<path>` before
    ///    the credential rule can fire on it, and the secret in the next token survives.
    #[test]
    fn credential_shapes_that_still_leak_are_a_documented_residual_not_a_claim() {
        const SECRET: &str = "sk-live-abc123";
        // If any of these starts passing the residual is PARTLY closed — update this test and
        // #140 rather than deleting the row, so the causes stay separable.
        for (cause, line) in [
            // (1) wrapper split
            (1, r#"{"Authorization":"Bearer sk-live-abc123"}"#),
            (1, "(Authorization=Bearer sk-live-abc123)"),
            (1, "env=Authorization:Bearer sk-live-abc123"),
            (1, "[Authorization:Bearer sk-live-abc123]"),
            (1, "hdr={Authorization: Bearer sk-live-abc123}"),
            // (2) detached-colon suffix — the whole line survives untouched
            (2, "Authorization :Bearer sk-live-abc123"),
            // (3) path absorption — the key is eaten by `<path>` before the rule sees it
            (3, "cannot write /tmp/Authorization:Bearer sk-live-abc123"),
        ] {
            assert!(
                scrub_engine_text(line).contains(SECRET),
                "cause-{cause} shape now SEALS — residual partly closed, update #140: {line}"
            );
        }
        // Cause (2) is the worst of the three: nothing at all is redacted, so the line does not
        // even look scrubbed. Pinned explicitly so that stays visible.
        assert_eq!(
            scrub_engine_text("Authorization :Bearer sk-live-abc123"),
            "Authorization :Bearer sk-live-abc123"
        );
        // Cause (3) DOES seal the path — it is the second-word secret that escapes.
        assert_eq!(
            scrub_engine_text("cannot write /tmp/Authorization:Bearer sk-live-abc123"),
            "cannot write <path> sk-live-abc123"
        );
        // The quoted-but-unwrapped form DOES seal: it stays one token, so the delimiter rule sees
        // it. This is the boundary between the two behaviours.
        let sealed = scrub_engine_text(r#""Authorization": "Bearer sk-live-abc123""#);
        assert!(!sealed.contains(SECRET), "{sealed}");
    }

    /// Idempotence has to hold across EVERY rule, because the seal boundary re-scrubs a string
    /// whose lines were already scrubbed individually and then joined. The header-form rule is the
    /// dangerous one: without its `<redacted>` lookahead it would eat the rest of the joined
    /// string on the second pass, destroying later lines' diagnostics.

    #[test]
    fn scrubbing_is_idempotent() {
        for line in [
            "at /Users/x/g.json TOKEN=abc host=h.internal",
            "Failed to connect to api.example.internal port 443",
            "dial host-dev.example.fail:443",
            "open ../../Users/operator/x failed",
            "AWS_SECRET_ACCESS_KEY: wJalrXUtnFEMIK7MDENGbPxRfiCY",
            "Authorization: Bearer sk-live-abc123",
            r#"{"AWS_SECRET_ACCESS_KEY":"wJal"}"#,
            "open /tmp/dump/AWS_SECRET_ACCESS_KEY=wJal",
        ] {
            // TRIPLE, not double: a rule that is stable on pass 2 can still move on pass 3 if it
            // rewrites its own output into a new shape. The seal boundary can re-scrub more than
            // once (source, join, seal), so two passes is not the real contract.
            let once = scrub_engine_text(line);
            let twice = scrub_engine_text(&once);
            assert_eq!(twice, once, "not idempotent at pass 2: {line}");
            assert_eq!(
                scrub_engine_text(&twice),
                twice,
                "not idempotent at pass 3: {line}"
            );
        }
    }

    /// The header-form rule must not swallow a joined tail on the second pass.
    #[test]
    fn header_form_redaction_does_not_eat_later_lines_on_reseal() {
        let joined = format!(
            "{} | {}",
            scrub_engine_text("Authorization: Bearer sk-live-abc123"),
            scrub_engine_text("CRITICAL: metal device init failed")
        );
        let resealed = scrub_reason_for_seal(&joined);
        assert!(
            resealed.contains("metal device init failed"),
            "re-sealing ate the later line: {resealed}"
        );
        assert!(!resealed.contains("sk-live-abc123"), "{resealed}");
    }

    #[test]
    fn sealed_reason_is_capped_and_scrubbed() {
        let long = format!("prefix {} /Users/x/tail.json", "A".repeat(8192));
        let sealed = scrub_reason_for_seal(&long);
        assert!(
            sealed.len() <= SEALED_REASON_BYTE_LIMIT,
            "sealed reason not capped: {} bytes",
            sealed.len()
        );
        assert!(sealed.starts_with("prefix "), "head lost: {sealed}");
        assert!(sealed.contains("<path>/tail.json"), "tail lost: {sealed}");
        assert!(!sealed.contains("/Users/x"), "path leaked: {sealed}");
    }

    /// Clipping must never split a UTF-8 character, whatever the cap lands on.
    #[test]
    fn clipping_respects_char_boundaries() {
        let text = "é".repeat(400); // 2 bytes each
        for cap in [1usize, 5, 11, 12, 13, 64, 799] {
            let clipped = clip_to_bytes(&text, cap);
            assert!(
                clipped.len() <= cap,
                "cap {cap} exceeded: {}",
                clipped.len()
            );
            // Round-trips as valid UTF-8 by construction (it is a String), so just assert it
            // is a prefix-shaped result and did not panic.
            assert!(clipped
                .chars()
                .all(|c| c == 'é' || "…[clipped]".contains(c)));
        }
    }
}
