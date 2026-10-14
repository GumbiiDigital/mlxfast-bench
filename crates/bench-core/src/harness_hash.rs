//! The TRUSTED-SIDE harness identity: a byte-compatible Rust port of the engine's `harnessHash()`.
//!
//! ## Why this lives in benchd (and must never come off the wire)
//!
//! `metrics.harness_hash` is a PUBLICATION-GATING value: the seam-3 overlay refuses to publish an
//! official score whose gates-score carries an empty or malformed harness identity
//! ([`crate::harness_hash`]'s consumer is `benchd::overlay::validate_gates`). Until F1 every
//! benchd-authored `gates-score.json` sealed `harness_hash = ""` — benchd had NO source for the
//! value — so no official score could publish.
//!
//! The value could NOT be sourced from the engine. The worker is PARTICIPANT-BUILT, so a
//! wire-reported hash would be attacker-controlled: a candidate could report the digest of a
//! harness it is not running. The computation therefore has to happen trusted-side, over the
//! workspace benchd itself drives — which is what this module does.
//!
//! ## The authoritative algorithm
//!
//! This is a byte-compatible port of the engines' own `harnessHash()`. There are now TWO reference
//! implementations, one per engine surface:
//!
//! * the MLX (Apple-silicon) engine's Swift
//!   `Sources/MLXFastTrustedHarness/QwenRuntimePreflight.swift` — `QwenRuntime.harnessHashRoots`,
//!   `QwenRuntime.harnessHashRootFiles(baseDirectory:)` and `QwenRuntime.harnessHash()`;
//! * the CUDA engine's equivalent, over that engine's own roster surface (`harness/`, `vllm/`,
//!   and `llama.cpp/` for a track that runs llama.cpp).
//!
//! Both compute the SAME algorithm over the roots that EXIST in their respective trees. This port is
//! byte-compatible with each: the same bytes go into SHA256, in the same order, so a benchd-computed
//! hash equals the one a reference implementation would compute over the same tree at the same
//! absolute location.
//!
//! Every semantic below was VERIFIED against a transcription of the reference symbols run against
//! synthetic trees (see the module tests for the resulting fixed-digest cross-implementation
//! regression vector and its provenance) — not inferred from reading:
//!
//! 1. **The hashed path strings are ABSOLUTE.** `URL(fileURLWithPath: root)` resolves a relative
//!    root against the process CWD and `.path` renders it absolute, so the digest covers the
//!    workspace's absolute location as well as its bytes. A harness hash is therefore an identity
//!    of *this tree at this path on this box*, NOT a portable content digest. (Surprising, but it
//!    is what the reference does, and byte-compatibility is the requirement.)
//! 2. **Production (`baseDirectory: nil`, CWD-relative) and an explicit fully-resolved
//!    `baseDirectory` produce the SAME digest** — verified equal on the same tree. That is the
//!    bridge this port stands on: [`harness_hash`] takes the workspace root explicitly, and
//!    [`harness_hash_of_current_dir`] passes `getcwd()` (always fully symlink-resolved, exactly
//!    what Foundation prepends), reproducing the reference's production path.
//! 3. **Ordering is over the FULL absolute path strings, globally** — `files.sorted()` in the
//!    reference, AFTER collection across all roots, so files from different roots interleave and the
//!    root declaration order does NOT affect the digest. ASCII ordering (`Sources/…` before
//!    `TASK.md` before `benchmark.json`) — Swift's `String` `<` and Rust's byte-wise `Ord` agree on
//!    ASCII, which every root name and every real workspace path is.
//! 4. **The hashed byte format is bare concatenation**: for each path in sorted order,
//!    `sha256.update(path.utf8)` then `sha256.update(file_bytes)`. NO separator, NO length prefix,
//!    NO trailing newline. Rendered as 64 lowercase hex characters.
//! 5. **An UNREADABLE file is skipped whole** — the reference's `guard let data = try?
//!    Data(contentsOf:) else { continue }` drops the path bytes too, not just the contents.
//! 6. **Hidden entries are skipped** (`.skipsHiddenFiles`): a dot-file inside a root, and a
//!    dot-directory's entire subtree, contribute nothing.
//! 7. **Symlinks inside a root directory are excluded entirely** — verified: a symlink to a file is
//!    not a regular file to `URLResourceValues` (no-follow), and the enumerator does not descend
//!    into a symlink to a directory. A symlinked root DIRECTORY therefore contributes zero files
//!    while still satisfying the existence probe (`fileExists` follows). A symlinked root FILE, by
//!    contrast, IS hashed with its target's bytes — the non-directory branch appends
//!    unconditionally, with no regular-file check, and the content read follows the link.
//! 8. **A MISSING root is LOGGED and SKIPPED, never a refusal.** The roster spans BOTH engine
//!    surfaces (MLX and CUDA), and neither engine has every root: the MLX tree has no `harness/` or
//!    `vllm/` or `llama.cpp/`, the CUDA tree has no `Package.swift`/`Sources`/`Tests`. Per David's
//!    ruling the hash is computed over what EXISTS — a root absent from disk contributes zero files (exactly as if
//!    it were an empty directory) and is noted on stderr so an operator can see what was covered.
//!    The SAME roster therefore serves both engines, and each hashes over its own surface.

use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::hash::hex_lower;

/// The FIXED root set the harness hash covers, spanning BOTH engine surfaces.
///
/// The first nine roots are the MLX engine's `QwenRuntime.harnessHashRoots`, transcribed in ORDER;
/// `harness`, `vllm`, and `llama.cpp` are CUDA engine harness surfaces, appended. The order does NOT
/// affect the digest (the collected paths are sorted globally before hashing), but it is pinned so
/// a root add/remove/reorder is a deliberate, reviewed act. A root that is absent on a given engine
/// is logged and skipped ([`harness_hash_root_files`]), so the same roster covers both engines and
/// each hashes over exactly the roots it actually has.
pub const HARNESS_HASH_ROOTS: [&str; 12] = [
    "Package.swift",
    "Sources",
    "Tests",
    "benchmark.json",
    "benchmark.sh",
    "setup.sh",
    "tools",
    "README.md",
    "TASK.md",
    "harness",
    "vllm",
    "llama.cpp",
];

/// Collect the regular files under [`HARNESS_HASH_ROOTS`], resolved against `workspace_root`, as
/// absolute path strings — the port of the reference `harnessHashRootFiles(baseDirectory:)`.
///
/// Returned UNSORTED, in the reference's collection order (root by root); [`harness_hash`] applies
/// the global sort. A root that is ABSENT from disk is logged to stderr and skipped — it contributes
/// zero files, the same as an empty directory would. The roster spans two engine surfaces and no
/// single engine has every root, so a missing root is the norm, not an error.
///
/// `workspace_root` is used AS GIVEN — never canonicalized — because the path strings it produces
/// go into the digest verbatim. Callers wanting the reference's production semantics must pass a
/// fully-resolved absolute root; [`harness_hash_of_current_dir`] does exactly that.
pub fn harness_hash_root_files(workspace_root: &Path) -> Vec<String> {
    root_files_over(workspace_root, &HARNESS_HASH_ROOTS)
}

/// [`harness_hash_root_files`] over an explicit roster. Production always passes
/// [`HARNESS_HASH_ROOTS`]; the tests pass an earlier roster to prove a roster addition does not
/// move the digest of a tree that lacks the added root.
fn root_files_over(workspace_root: &Path, roots: &[&str]) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    for &root in roots {
        let path = workspace_root.join(root);
        // `FileManager.fileExists(atPath:isDirectory:)` — FOLLOWS symlinks, so a symlinked root
        // satisfies the probe. An absent root (or a broken symlink) is LOGGED and SKIPPED: the
        // roster spans both engine surfaces, so a root the current engine does not carry
        // contributes nothing rather than aborting the hash.
        let meta = match fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => {
                eprintln!("harnessHash root absent, skipped: {root}");
                continue;
            }
        };
        if meta.is_dir() {
            // The reference `FileManager.enumerator(at:)` does NOT descend into a symlink to a
            // directory, so a symlinked root directory contributes ZERO files while still passing
            // the probe above (verified against the reference). `walk` reproduces that by testing
            // the no-follow type of every directory it is about to enter, this root included.
            walk_directory(&path, &mut files);
        } else {
            // The reference's non-directory branch appends UNCONDITIONALLY — no regular-file check
            // — so a symlinked root file is included here and hashed with its target's bytes (the
            // content read follows the link).
            files.push(path_string(&path));
        }
    }
    files
}

/// Recursively append the regular files under `dir`, mirroring
/// `FileManager.enumerator(at:includingPropertiesForKeys:[.isRegularFileKey], options:
/// [.skipsHiddenFiles])` + the `values?.isRegularFile == true` filter.
///
/// Three reference behaviors are reproduced exactly:
/// * a directory that cannot be enumerated is SKIPPED, not an error (the reference's
///   `guard let enumerator … else { continue }`);
/// * hidden entries — leading `.` — are skipped, and a hidden directory's whole subtree with them;
/// * symlinks are excluded at every level: `file_type()` comes from `readdir` and does not follow,
///   so a symlink is neither `is_file()` nor `is_dir()` and falls through both arms.
fn walk_directory(dir: &Path, files: &mut Vec<String>) {
    // No-follow type check on the directory itself: entering through a symlink is what the
    // reference enumerator refuses to do.
    match fs::symlink_metadata(dir) {
        Ok(md) if md.is_dir() => {}
        _ => return,
    }
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        // Not enumerable → skip this subtree. This is the reference's `continue`.
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        // `.skipsHiddenFiles`. macOS also honors the Finder "hidden" file flag; a leading dot is
        // the form that occurs in a source workspace and the only one reproduced here.
        if name.to_string_lossy().starts_with('.') {
            continue;
        }
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            walk_directory(&path, files);
        } else if file_type.is_file() {
            files.push(path_string(&path));
        }
        // else: symlink (or a socket/fifo/device) — excluded, as `isRegularFile` excludes it.
    }
}

/// The path string that goes INTO the digest. The reference hashes `Data(path.utf8)` off `URL.path`,
/// a `String`; a non-UTF-8 byte path cannot be that string, and lossy rendering keeps this total
/// rather than dropping a file silently.
fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// The DIGEST half of the algorithm, isolated from the filesystem: sort the collected absolute
/// paths, then for each one that reads, hash the path's UTF-8 bytes immediately followed by the
/// file's bytes.
///
/// `read` is the content source. A `None` from it is the reference's unreadable-file `continue`:
/// the path bytes are dropped WITH the contents, not hashed alone.
///
/// TEST-ONLY sugar over [`harness_hash_of_sorted`]: it exists so the byte format and the ordering
/// rule are testable with in-memory contents at fixed absolute path strings — which is what makes
/// a machine-independent cross-implementation vector possible at all, given that the digest covers
/// absolute paths. Production sorts once in [`harness_hash`] and folds through
/// [`harness_hash_of_sorted`], so both still hash through exactly the same loop.
#[cfg(test)]
fn harness_hash_over<F>(mut paths: Vec<String>, read: F) -> String
where
    F: FnMut(&str) -> Option<Vec<u8>>,
{
    // `files.sorted()` — the GLOBAL sort over full absolute path strings, after collection across
    // all roots. Rust's byte-wise `Ord` agrees with Swift's `String` ordering on ASCII.
    paths.sort();
    harness_hash_of_sorted(&paths, read)
}

/// The fold itself, over paths that are ALREADY in `files.sorted()` order.
///
/// Split from [`harness_hash_over`] so the production path can sort ONCE and hand the SAME slice
/// to both the read-ahead thread and this fold — see [`harness_hash_read_ahead`]. Having one
/// sorted list rather than two is what makes the prefetch safe by construction: the reader walks
/// exactly the sequence this loop consumes, so the digest cannot depend on two sorts agreeing.
fn harness_hash_of_sorted<F>(paths: &[String], mut read: F) -> String
where
    F: FnMut(&str) -> Option<Vec<u8>>,
{
    let mut hasher = Sha256::new();
    for path in paths {
        let Some(bytes) = read(path) else {
            continue;
        };
        hasher.update(path.as_bytes());
        hasher.update(&bytes);
    }
    hex_lower(&hasher.finalize())
}

/// How many files the read-ahead thread may hold in front of the hasher. The fold is strictly
/// single-threaded — update ORDER is the digest — so reading cannot be parallelised; it can only
/// be moved OFF the hashing thread, which is what this does. The bound is a file COUNT, and it is
/// small: the harness roster is a source surface, and the point is only to keep the next file's
/// bytes ready, not to buffer the tree.
const HARNESS_HASH_READ_AHEAD: usize = 4;

/// [`harness_hash_of_sorted`] with the file reads moved onto a second thread that runs up to
/// [`HARNESS_HASH_READ_AHEAD`] files ahead of the hasher.
///
/// The digest is unchanged, and unchangeable by this: the hasher still consumes `paths` in sorted
/// order on ONE thread, and the reader emits exactly that sequence over a bounded channel, so the
/// i-th value the fold receives is the i-th file's contents — the same `Option<Vec<u8>>` the old
/// inline `fs::read(p).ok()` produced, including `None` for an unreadable file (the reference's
/// skip-the-path-too behaviour).
fn harness_hash_read_ahead(paths: &[String]) -> String {
    let (tx, rx) = std::sync::mpsc::sync_channel::<Option<Vec<u8>>>(HARNESS_HASH_READ_AHEAD);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            for path in paths {
                // A closed receiver means the fold finished early (it cannot, today) — stop
                // rather than read the rest of the tree for nobody.
                if tx.send(fs::read(path).ok()).is_err() {
                    return;
                }
            }
        });
        // `recv` can only fail once the reader has finished AND every queued item is drained,
        // which cannot happen before the loop ends: the reader sends exactly one item per path.
        // Treating that impossible case as `None` keeps this total.
        harness_hash_of_sorted(paths, |_| rx.recv().unwrap_or(None))
    })
}

/// The harness identity of the workspace rooted at `workspace_root`: 64 lowercase hex characters,
/// byte-compatible with the engine's `harnessHash()` over the same tree at the same absolute path.
///
/// Hashes over the roster roots that EXIST — a root absent from disk is logged and skipped
/// ([`harness_hash_root_files`]), so the same roster serves both the MLX and CUDA engine surfaces
/// and this function never refuses on a missing root.
pub fn harness_hash(workspace_root: &Path) -> String {
    let mut files = harness_hash_root_files(workspace_root);
    // The ONE sort. Both the read-ahead thread and the fold walk this exact slice.
    files.sort();
    harness_hash_read_ahead(&files)
}

/// [`harness_hash`] over the process's current working directory — the reference's PRODUCTION
/// path, where the roots resolve CWD-relative.
///
/// `getcwd()` returns a fully symlink-resolved absolute path, which is precisely what Foundation
/// prepends to a relative `URL(fileURLWithPath:)`, so this reproduces `harnessHash()`'s own digest.
/// Verified: the reference's CWD path and an explicit fully-resolved base directory agree. The only
/// failure mode is an unresolvable CWD (the roots themselves never refuse), so this returns `Err`
/// solely for that.
pub fn harness_hash_of_current_dir() -> Result<String, String> {
    let cwd: PathBuf = std::env::current_dir()
        .map_err(|e| format!("harnessHash: cannot resolve the current working directory: {e}"))?;
    Ok(harness_hash(&cwd))
}

/// Whether `s` is the shape a harness hash must have: 64 LOWERCASE hex characters. The same grid
/// the overlay's gates-input refusal enforces; exposed here so a producer can assert what it sealed
/// against the identical predicate its consumer will apply.
pub fn is_well_formed_harness_hash(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    // -----------------------------------------------------------------------
    // The CROSS-IMPLEMENTATION vector and its provenance
    // -----------------------------------------------------------------------

    /// The absolute base the cross-implementation vector was computed at. It is part of the digest
    /// (see the module doc, point 1), so it is pinned as data rather than discovered at test time —
    /// which is what makes the vector reproducible on any machine and any OS.
    const VECTOR_BASE: &str = "/private/tmp/benchd-hh-vector-v1";

    /// The SYNTHETIC fixture tree the vector covers, as `(path-relative-to-VECTOR_BASE, bytes)`.
    ///
    /// This is an MLX-SHAPED tree: it has the nine MLX roots and NONE of the CUDA roots (`harness/`,
    /// `vllm/`, `llama.cpp/`). It is the exact set the reference COLLECTED from the on-disk tree — note what is
    /// absent: `Sources/.hidden.swift` and `Sources/.hiddendir/C.swift` existed on disk and are not
    /// here (`.skipsHiddenFiles`), and `Sources/empty/` contributed nothing. Deliberately listed in
    /// NON-sorted order so a port that forgets the global sort cannot pass.
    ///
    /// Because it has no `harness/` or `vllm/`, this same fixture is the DIGEST-STABILITY witness:
    /// [`VECTOR_HASH`] was pinned BEFORE `harness`/`vllm` joined the roster, so its still-matching
    /// here proves that widening the roster + the missing-root skip do not move an MLX tree's digest.
    const VECTOR_FILES: [(&str, &str); 11] = [
        ("Package.swift", "package-swift-bytes\n"),
        ("Sources/A.swift", "sources-a\n"),
        ("Sources/nested/B.swift", "sources-nested-b\n"),
        ("Tests/T.swift", "tests-t\n"),
        ("benchmark.json", "{\"k\":1}\n"),
        ("benchmark.sh", "benchmark-sh\n"),
        ("setup.sh", "setup-sh\n"),
        ("tools/t.sh", "tools-t\n"),
        ("tools/sub/u.sh", "tools-sub-u\n"),
        ("README.md", "readme\n"),
        ("TASK.md", "task\n"),
    ];

    /// The vector: what the ENGINE'S reference `harnessHash()` produces over [`VECTOR_FILES`] laid
    /// out at [`VECTOR_BASE`].
    ///
    /// PROVENANCE — this number was not copied from anywhere; it was produced for this test:
    /// `QwenRuntime.harnessHashRoots`, `.harnessHashRootFiles(baseDirectory:)` and `.harnessHash()`
    /// were transcribed VERBATIM out of the MLX engine's
    /// `Sources/MLXFastTrustedHarness/QwenRuntimePreflight.swift` into a throwaway Swift script,
    /// compiled with the local Xcode toolchain (`DEVELOPER_DIR=/Applications/Xcode.app/Contents/
    /// Developer swiftc`, Apple Swift 6.3.3), and run against the tree above built at `VECTOR_BASE`.
    /// The script computed the same digest by BOTH reference paths — the production CWD path
    /// (`chdir(VECTOR_BASE)`, `baseDirectory: nil`) and the explicit-base test seam — which is the
    /// evidence for the claim in the module doc that [`harness_hash_of_current_dir`] reproduces
    /// production `harnessHash()`.
    ///
    /// It is PINNED as a fixed-digest regression: the value is over a nine-root (MLX-shaped) tree
    /// and did NOT change when `harness`/`vllm` were added to the roster (those roots are absent
    /// from this fixture, so they are skipped and contribute nothing).
    const VECTOR_HASH: &str = "edd0ac485d9e684770dd5866183bf4fee0166c25c73d989f240ed5f591fdbcc4";

    /// MUTATION PROOF (a) — the byte format AND the ordering rule, against the reference number.
    ///
    /// Reds if the port reorders the files (the entries are fed unsorted, so dropping
    /// `paths.sort()` changes the digest), or alters the hashed byte format in any way: a
    /// separator between path and contents, a length prefix, hashing contents before the path,
    /// omitting the path, or upper-case hex rendering.
    #[test]
    fn rust_harness_hash_matches_the_cross_implementation_vector() {
        let mut contents: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut paths: Vec<String> = Vec::new();
        for (rel, body) in VECTOR_FILES {
            let abs = format!("{VECTOR_BASE}/{rel}");
            contents.insert(abs.clone(), body.as_bytes().to_vec());
            paths.push(abs);
        }
        let got = harness_hash_over(paths, |p| contents.get(p).cloned());
        assert_eq!(
            got, VECTOR_HASH,
            "the Rust port must be BYTE-COMPATIBLE with the engine's harnessHash()"
        );
        assert!(is_well_formed_harness_hash(&got));
    }

    /// The unreadable-file rule (module doc point 5): the path bytes are dropped WITH the contents.
    /// A port that hashed the path and then skipped only the body would produce a different digest
    /// from the reference on any tree with one unreadable file.
    #[test]
    fn an_unreadable_file_contributes_neither_its_path_nor_its_bytes() {
        let present = harness_hash_over(vec!["/w/a".to_string()], |p| {
            (p == "/w/a").then(|| b"A".to_vec())
        });
        let with_unreadable =
            harness_hash_over(vec!["/w/a".to_string(), "/w/b".to_string()], |p| {
                (p == "/w/a").then(|| b"A".to_vec())
            });
        assert_eq!(
            present, with_unreadable,
            "an unreadable file must be skipped WHOLE — path bytes included"
        );
    }

    /// The twelve roots, in the pinned order: the nine MLX roots (with `benchmark.sh` at index 4,
    /// the property the engine's own `HarnessHashRootSetTests.swift` pins) followed by the CUDA
    /// engine roots. A root added, removed or reordered here changes what benchd hashes and must be
    /// a deliberate, reviewed act.
    #[test]
    fn the_root_set_is_the_two_engine_roster_in_order() {
        assert_eq!(
            HARNESS_HASH_ROOTS,
            [
                "Package.swift",
                "Sources",
                "Tests",
                "benchmark.json",
                "benchmark.sh",
                "setup.sh",
                "tools",
                "README.md",
                "TASK.md",
                "harness",
                "vllm",
                "llama.cpp",
            ]
        );
        assert_eq!(HARNESS_HASH_ROOTS[4], "benchmark.sh");
        assert_eq!(HARNESS_HASH_ROOTS[9], "harness");
        assert_eq!(HARNESS_HASH_ROOTS[10], "vllm");
        assert_eq!(HARNESS_HASH_ROOTS[11], "llama.cpp");
        assert_eq!(HARNESS_HASH_ROOTS.len(), 12);
    }

    // -----------------------------------------------------------------------
    // Filesystem behavior (prefix-independent: built under a real temp dir)
    // -----------------------------------------------------------------------

    /// The read-ahead path's `None` arm, over a REAL unreadable file rather than an in-memory
    /// stub: the reference drops the path bytes WITH the contents, so an unreadable file must hash
    /// exactly as if it were not there. The in-memory twin of this
    /// (`an_unreadable_file_contributes_neither_its_path_nor_its_bytes`) cannot reach the reader
    /// thread; this one does. ONE root throughout — the digest covers absolute paths, so two trees
    /// at two locations are not comparable.
    #[test]
    fn an_unreadable_file_on_disk_hashes_as_if_absent() {
        use std::os::unix::fs::PermissionsExt;
        let root = tmp_root("unreadable");
        build_synthetic_tree(&root);
        let without = harness_hash(&root);

        let blocked = root.join("Sources/nested/Blocked.swift");
        write(&blocked, "secret\n");
        fs::set_permissions(&blocked, fs::Permissions::from_mode(0o000)).expect("chmod 000");
        let with_unreadable = harness_hash(&root);

        assert_eq!(
            with_unreadable, without,
            "an unreadable file must contribute neither its path nor its bytes"
        );

        // Readable again, it DOES move the digest — otherwise the assertion above would also pass
        // for a walk that never saw the file at all.
        fs::set_permissions(&blocked, fs::Permissions::from_mode(0o644)).expect("chmod 644");
        assert_ne!(harness_hash(&root), without);

        let _ = fs::remove_dir_all(&root);
    }

    fn tmp_root(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("benchd-hh-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn write(path: &Path, body: &str) {
        fs::create_dir_all(path.parent().expect("file has a parent")).expect("mkdir -p");
        fs::write(path, body).expect("write fixture file");
    }

    /// Build the SAME synthetic tree the cross-implementation vector covers, at `root` — an
    /// MLX-shaped tree (nine roots, no `harness/` or `vllm/`), including the entries the reference
    /// must SKIP (a hidden file, a hidden directory's subtree, an empty directory), so the
    /// collection rules are exercised, not just the digest.
    fn build_synthetic_tree(root: &Path) {
        for (rel, body) in VECTOR_FILES {
            write(&root.join(rel), body);
        }
        write(&root.join("Sources/.hidden.swift"), "sources-hidden-file\n");
        write(
            &root.join("Sources/.hiddendir/C.swift"),
            "sources-hidden-dir-c\n",
        );
        fs::create_dir_all(root.join("Sources/empty")).expect("mkdir empty");
    }

    fn sorted_relative(root: &Path, files: &[String]) -> Vec<String> {
        let prefix = format!("{}/", root.display());
        let mut rels: Vec<String> = files
            .iter()
            .map(|f| {
                f.strip_prefix(&prefix)
                    .unwrap_or_else(|| panic!("collected path {f} is not under {prefix}"))
                    .to_string()
            })
            .collect();
        rels.sort();
        rels
    }

    /// The COLLECTION half against the reference: the exact file set the transcription collected
    /// from this MLX-shaped tree — hidden file skipped, hidden directory's subtree skipped, empty
    /// directory contributing nothing, and the two absent CUDA roots skipped.
    #[test]
    fn collection_matches_the_reference_file_set() {
        let root = tmp_root("collect");
        build_synthetic_tree(&root);
        let files = harness_hash_root_files(&root);
        let mut expected: Vec<String> = VECTOR_FILES.iter().map(|(r, _)| r.to_string()).collect();
        expected.sort();
        assert_eq!(sorted_relative(&root, &files), expected);
        let _ = fs::remove_dir_all(&root);
    }

    /// STABILITY — hashing the same unchanged tree twice yields the same digest.
    #[test]
    fn hashing_the_same_tree_twice_is_stable() {
        let root = tmp_root("stable");
        build_synthetic_tree(&root);
        let a = harness_hash(&root);
        let b = harness_hash(&root);
        assert_eq!(
            a, b,
            "the harness hash must be stable over an unchanged tree"
        );
        assert!(is_well_formed_harness_hash(&a));
        let _ = fs::remove_dir_all(&root);
    }

    /// SENSITIVITY — adding a file under a root MOVES the hash; removing it RETURNS the hash.
    /// Also covers content sensitivity (editing a hashed file moves it too).
    #[test]
    fn adding_a_file_under_a_root_moves_the_hash_and_removing_it_returns() {
        let root = tmp_root("sensitive");
        build_synthetic_tree(&root);
        let before = harness_hash(&root);

        let added = root.join("Sources/nested/Added.swift");
        write(&added, "added\n");
        let with_added = harness_hash(&root);
        assert_ne!(
            before, with_added,
            "a new file under a hashed root must move the harness hash"
        );

        fs::remove_file(&added).expect("remove added file");
        assert_eq!(
            before,
            harness_hash(&root),
            "removing the added file must return the original harness hash"
        );

        fs::write(root.join("TASK.md"), "task-edited\n").expect("edit");
        assert_ne!(
            before,
            harness_hash(&root),
            "editing a hashed file's bytes must move the harness hash"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// A HIDDEN file added under a root does NOT move the hash — the reference skips it, so a port
    /// that walked hidden entries would diverge on any workspace with a `.DS_Store` in `Sources/`.
    #[test]
    fn a_hidden_file_added_under_a_root_does_not_move_the_hash() {
        let root = tmp_root("hidden");
        build_synthetic_tree(&root);
        let before = harness_hash(&root);
        write(&root.join("tools/.DS_Store"), "junk\n");
        write(&root.join("Tests/.hiddendir/deep.swift"), "deep\n");
        assert_eq!(
            before,
            harness_hash(&root),
            "hidden entries must contribute nothing (.skipsHiddenFiles)"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// MISSING-ROOT SKIP (module doc point 8, David's ruling) — a workspace missing a root HASHES
    /// over the roots it has instead of refusing. Removing a root drops that root's files (so the
    /// digest moves, exactly as if the files were deleted) but yields a well-formed digest, never an
    /// error. Verified for EVERY roster root, one at a time.
    #[test]
    fn a_missing_root_is_logged_and_skipped_for_every_root() {
        for root_name in HARNESS_HASH_ROOTS {
            let root = tmp_root(&format!("missing-{}", root_name.replace(['.', '/'], "_")));
            build_synthetic_tree(&root);
            // `full` and `got` are computed at the SAME absolute base — the digest covers absolute
            // paths, so a comparison is only meaningful when both legs share a root.
            let full = harness_hash(&root);
            let target = root.join(root_name);
            // The CUDA roots are absent from the MLX-shaped fixture already; the nine MLX roots
            // are removed here. Either way the hash must be produced, not refused.
            if target.is_dir() {
                fs::remove_dir_all(&target).expect("remove root dir");
            } else if target.exists() {
                fs::remove_file(&target).expect("remove root file");
            }
            let got = harness_hash(&root);
            assert!(
                is_well_formed_harness_hash(&got),
                "a missing root must still yield a well-formed digest, not refuse: {root_name}"
            );
            // A root that was PRESENT and carried files must move the digest when dropped; an
            // already-absent CUDA root leaves the digest unchanged.
            if ["harness", "vllm", "llama.cpp"].contains(&root_name) {
                assert_eq!(
                    got, full,
                    "dropping an already-absent CUDA root cannot change the digest: {root_name}"
                );
            } else {
                assert_ne!(
                    got, full,
                    "dropping a populated MLX root must move the digest: {root_name}"
                );
            }
            let _ = fs::remove_dir_all(&root);
        }
    }

    /// DIGEST STABILITY — an MLX-shaped tree (nine roots present, no `harness/` or `vllm/`) hashes
    /// to EXACTLY [`VECTOR_HASH`], the value pinned BEFORE `harness`/`vllm` joined the roster.
    /// This is the load-bearing proof that widening the roster + the missing-root skip did NOT move
    /// the MLX digest: the two new roots are absent, so they are skipped and contribute nothing, and
    /// the sorted file set — hence the digest — is identical to what the nine-root roster produced.
    /// Adding EMPTY `harness/` and `vllm/` directories (present but contributing no files) must not
    /// move it either.
    #[test]
    fn an_mlx_tree_is_stable_under_the_widened_roster() {
        // In-memory over the exact nine-root fixture: byte-identical to the pre-change vector.
        let mut contents: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut paths: Vec<String> = Vec::new();
        for (rel, body) in VECTOR_FILES {
            let abs = format!("{VECTOR_BASE}/{rel}");
            contents.insert(abs.clone(), body.as_bytes().to_vec());
            paths.push(abs);
        }
        assert_eq!(
            harness_hash_over(paths, |p| contents.get(p).cloned()),
            VECTOR_HASH,
            "an MLX-shaped tree must hash to the pre-widening pinned digest — the roster addition \
             and the missing-root skip must not move it"
        );

        // On-disk: an MLX tree, then the same tree with EMPTY CUDA roots present. Both must equal
        // each other and be well-formed (present-but-empty roots contribute no files).
        let root = tmp_root("mlx-stable");
        build_synthetic_tree(&root);
        let mlx = harness_hash(&root);
        assert!(is_well_formed_harness_hash(&mlx));
        fs::create_dir_all(root.join("harness")).expect("mkdir harness");
        fs::create_dir_all(root.join("vllm")).expect("mkdir vllm");
        fs::create_dir_all(root.join("llama.cpp")).expect("mkdir llama.cpp");
        assert_eq!(
            mlx,
            harness_hash(&root),
            "empty harness/, vllm/, and llama.cpp/ roots contribute no files and must not move the digest"
        );

        write(&root.join("llama.cpp/CMakeLists.txt"), "cmake\n");
        assert_ne!(
            mlx,
            harness_hash(&root),
            "a populated llama.cpp/ root must move the digest"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// A CUDA-SHAPED tree — the shared roots plus `harness/` and `vllm/`, and NONE of
    /// `Package.swift`/`Sources`/`Tests` — hashes over `harness/`+`vllm/` (they ARE collected) while
    /// the absent Swift roots are skipped. Populating `harness/` or `vllm/` moves the digest.
    #[test]
    fn a_cuda_tree_hashes_harness_and_vllm_and_skips_the_swift_roots() {
        let root = tmp_root("cuda");
        for rel in [
            "benchmark.json",
            "benchmark.sh",
            "setup.sh",
            "tools/t.sh",
            "README.md",
            "TASK.md",
            "harness/protocol-adapter/src/vllm_backend.rs",
            "vllm/csrc/kernel.cu",
        ] {
            write(&root.join(rel), rel);
        }
        let files = harness_hash_root_files(&root);
        let rels = sorted_relative(&root, &files);
        assert!(
            rels.iter().any(|f| f.starts_with("harness/")),
            "a CUDA tree's harness/ must be collected: {rels:?}"
        );
        assert!(
            rels.iter().any(|f| f.starts_with("vllm/")),
            "a CUDA tree's vllm/ must be collected: {rels:?}"
        );
        assert!(
            !rels.iter().any(|f| f.starts_with("Sources/")
                || f.starts_with("Tests/")
                || f == "Package.swift"),
            "absent Swift roots must be skipped, not present: {rels:?}"
        );

        let before = harness_hash(&root);
        write(
            &root.join("harness/protocol-adapter/src/added.rs"),
            "added\n",
        );
        assert_ne!(
            before,
            harness_hash(&root),
            "a new file under harness/ must move a CUDA tree's digest"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// The roster as it stood BEFORE `llama.cpp` joined it, transcribed as data so the comparison
    /// below is against the old roster itself, not a slice of the new one.
    const ROSTER_BEFORE_LLAMA_CPP: [&str; 11] = [
        "Package.swift",
        "Sources",
        "Tests",
        "benchmark.json",
        "benchmark.sh",
        "setup.sh",
        "tools",
        "README.md",
        "TASK.md",
        "harness",
        "vllm",
    ];

    /// The digest of `root` over an explicit roster, by the production sort and fold.
    fn harness_hash_over_roster(root: &Path, roots: &[&str]) -> String {
        let mut files = root_files_over(root, roots);
        files.sort();
        harness_hash_read_ahead(&files)
    }

    /// DIGEST STABILITY under the `llama.cpp` addition: a tree WITHOUT `llama.cpp/` hashes to the
    /// same digest under the old eleven-root roster and the current roster. Checked for an
    /// MLX-shaped tree and for a vLLM-shaped CUDA tree (`harness/` + `vllm/`, no Swift roots, the
    /// Nemotron CUDA surface), so neither digest moves.
    #[test]
    fn a_tree_without_llama_cpp_hashes_as_under_the_old_roster() {
        let mlx = tmp_root("no-llama-mlx");
        build_synthetic_tree(&mlx);
        assert_eq!(
            harness_hash(&mlx),
            harness_hash_over_roster(&mlx, &ROSTER_BEFORE_LLAMA_CPP),
            "an MLX tree's digest must not move when llama.cpp joins the roster"
        );
        let _ = fs::remove_dir_all(&mlx);

        let cuda = tmp_root("no-llama-cuda");
        for rel in [
            "benchmark.json",
            "benchmark.sh",
            "setup.sh",
            "tools/t.sh",
            "README.md",
            "TASK.md",
            "harness/protocol-adapter/src/vllm_backend.rs",
            "vllm/csrc/kernel.cu",
        ] {
            write(&cuda.join(rel), rel);
        }
        assert_eq!(
            harness_hash(&cuda),
            harness_hash_over_roster(&cuda, &ROSTER_BEFORE_LLAMA_CPP),
            "a vLLM CUDA tree's digest must not move when llama.cpp joins the roster"
        );
        let _ = fs::remove_dir_all(&cuda);
    }

    /// A llama.cpp-shaped CUDA tree (`harness/` + `llama.cpp/`, no `vllm/`, no Swift roots) hashes
    /// over `llama.cpp/`: its files are collected, the digest differs from the old roster's, and a
    /// new or changed file under `llama.cpp/` moves the digest.
    #[test]
    fn a_llama_cpp_file_moves_the_digest() {
        let root = tmp_root("llama-cpp");
        for rel in [
            "benchmark.json",
            "benchmark.sh",
            "setup.sh",
            "tools/t.sh",
            "README.md",
            "TASK.md",
            "harness/protocol-adapter/src/backend.rs",
            "llama.cpp/CMakeLists.txt",
            "llama.cpp/src/llama.cpp",
        ] {
            write(&root.join(rel), rel);
        }
        let rels = sorted_relative(&root, &harness_hash_root_files(&root));
        assert!(
            rels.iter().any(|f| f == "llama.cpp/src/llama.cpp"),
            "a CUDA tree's llama.cpp/ must be collected: {rels:?}"
        );
        let with = harness_hash(&root);
        assert_ne!(
            with,
            harness_hash_over_roster(&root, &ROSTER_BEFORE_LLAMA_CPP),
            "llama.cpp/ files must count under the current roster"
        );

        write(&root.join("llama.cpp/ggml/src/added.cu"), "added\n");
        let added = harness_hash(&root);
        assert_ne!(
            with, added,
            "a new file under llama.cpp/ must move the digest"
        );

        write(&root.join("llama.cpp/CMakeLists.txt"), "changed\n");
        assert_ne!(
            added,
            harness_hash(&root),
            "a changed file under llama.cpp/ must move the digest"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// An EMPTY root directory contributes nothing — the same outcome as a missing root, which is
    /// now also a skip. Kept to pin that a present-but-empty root and an absent one agree.
    #[test]
    fn an_empty_root_directory_contributes_nothing() {
        let root = tmp_root("emptyroot");
        build_synthetic_tree(&root);
        fs::remove_dir_all(root.join("tools")).expect("drop tools");
        fs::create_dir_all(root.join("tools")).expect("recreate empty tools");
        let files = harness_hash_root_files(&root);
        assert!(
            !sorted_relative(&root, &files)
                .iter()
                .any(|f| f.starts_with("tools/")),
            "an empty root contributes no files"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// SYMLINK EXCLUSION (module doc point 7), verified against the reference: a symlink to a file
    /// and a symlink to a directory inside a hashed root are both excluded, so neither moves the
    /// hash.
    #[cfg(unix)]
    #[test]
    fn symlinks_inside_a_root_are_excluded() {
        let root = tmp_root("symlink");
        build_synthetic_tree(&root);
        let before = harness_hash(&root);

        let real_dir = root.join("outside/subdir");
        write(&real_dir.join("R.swift"), "real-file\n");
        std::os::unix::fs::symlink(&real_dir, root.join("Sources/linkdir")).expect("symlink dir");
        std::os::unix::fs::symlink(
            root.join("Package.swift"),
            root.join("tools/linkfile.swift"),
        )
        .expect("symlink file");
        std::os::unix::fs::symlink(root.join("nowhere"), root.join("Tests/brokenlink"))
            .expect("symlink broken");

        assert_eq!(
            before,
            harness_hash(&root),
            "symlinks inside a hashed root are excluded entirely"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// A symlinked root DIRECTORY passes the existence probe and contributes ZERO files — the
    /// reference's `fileExists` follows, its enumerator does not. Pinned because the two halves
    /// disagreeing is exactly the kind of quirk a port silently "fixes" into a divergence.
    #[cfg(unix)]
    #[test]
    fn a_symlinked_root_directory_is_present_but_contributes_nothing() {
        let root = tmp_root("rootlink");
        build_synthetic_tree(&root);
        fs::rename(root.join("tools"), root.join("tools-real")).expect("rename");
        std::os::unix::fs::symlink(root.join("tools-real"), root.join("tools")).expect("symlink");
        let files = harness_hash_root_files(&root);
        assert!(
            !sorted_relative(&root, &files)
                .iter()
                .any(|f| f.starts_with("tools")),
            "a symlinked root directory contributes no files"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// The cross-implementation vector END TO END, over a REAL tree on disk — collection and
    /// digest together, not the digest half alone. Also the on-disk digest-stability witness: the
    /// tree is MLX-shaped (no `harness/`/`vllm/`) and must reproduce the pre-widening [`VECTOR_HASH`].
    ///
    /// The digest covers absolute paths, so this can only run where the fixture can be built at
    /// exactly [`VECTOR_BASE`]; `/private/tmp` is macOS's real `/tmp`, which is where the
    /// transcription computed [`VECTOR_HASH`]. Gated to macOS because that is precisely the scope
    /// of the claim: byte-compatibility with the engine's Swift `harnessHash()`, which is a
    /// macOS-only binary running on a macOS ranked box. The portable half of the proof
    /// (`rust_harness_hash_matches_the_cross_implementation_vector` +
    /// `collection_matches_the_reference_file_set`) runs everywhere.
    #[cfg(target_os = "macos")]
    #[test]
    fn end_to_end_over_a_real_tree_matches_the_vector() {
        let root = PathBuf::from(VECTOR_BASE);
        let _ = fs::remove_dir_all(&root);
        build_synthetic_tree(&root);
        let got = harness_hash(&root);
        assert_eq!(
            got, VECTOR_HASH,
            "walking the real tree must reproduce the pinned digest, collection and format together"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// The production entry point agrees with the explicit-root one over the same tree. This is the
    /// Rust side of the reference equality the vector's provenance records (CWD path == explicit
    /// fully-resolved base). `current_dir()` is process-global, so this asserts the composition
    /// rather than mutating it.
    #[test]
    fn current_dir_entry_point_hashes_the_current_directory() {
        let cwd = std::env::current_dir().expect("cwd");
        // Whatever the crate's CWD is, the two entry points must agree.
        assert_eq!(harness_hash_of_current_dir(), Ok(harness_hash(&cwd)));
    }

    #[test]
    fn well_formed_predicate_is_64_lowercase_hex() {
        assert!(is_well_formed_harness_hash(&"a".repeat(64)));
        assert!(is_well_formed_harness_hash(VECTOR_HASH));
        assert!(!is_well_formed_harness_hash(""));
        assert!(!is_well_formed_harness_hash(&"a".repeat(63)));
        assert!(!is_well_formed_harness_hash(&"a".repeat(65)));
        assert!(!is_well_formed_harness_hash(&"A".repeat(64)));
        assert!(!is_well_formed_harness_hash(&"g".repeat(64)));
    }
}
