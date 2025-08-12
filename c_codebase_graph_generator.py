#!/usr/bin/env python3
"""
cgraph.py — Build an undirected dependency graph for a C codebase using libclang.

Nodes:
  - id: <module.path>.<symbol>  (e.g., src.utils.math.add::function)
  - attrs:
      kind: "function" | "struct" | "global" | "typedef"
      code: source snippet (definition)
      file: relative source path
      usr:  Clang USR for stable identity
      ...

Edges (undirected):
  - function <-> function      (calls or function address usage)
  - function <-> struct        (types in params/return/locals, member use)
  - function <-> typedef       (typedef appears in types)
  - function <-> global        (global var reference)
  - struct  <-> struct         (field types)
  - struct  <-> typedef        (underlying/field typedefs)
  - typedef <-> struct         (aliasing underlying record)

Only in-project definitions become nodes/edge endpoints.
"""

import argparse
import glob
import hashlib
import html
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from clang import cindex
from clang.cindex import Cursor, CursorKind, TypeKind
from matplotlib.patches import Patch
from pyvis.network import Network

# ------------------------- libclang discovery -------------------------


def init_libclang():
    # Prefer explicit env var; otherwise let cindex try defaults.
    lib = os.environ.get("CLANG_LIBRARY_FILE")
    if lib and os.path.exists(lib):
        cindex.Config.set_library_file(lib)


def guess_macos_sys_includes():
    paths = []
    for pat in (
        "/opt/homebrew/opt/llvm/lib/clang/*/include",
        "/usr/local/opt/llvm/lib/clang/*/include",
        "/Library/Developer/CommandLineTools/usr/lib/clang/*/include",
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/*/include",
    ):
        hits = sorted(glob.glob(pat))
        if hits:
            paths.append(hits[-1])
            break
    for sdk in (
        "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk",
    ):
        if os.path.isdir(sdk):
            return sum((["-isystem", p] for p in paths), []) + ["-isysroot", sdk]
    return sum((["-isystem", p] for p in paths), [])


# ------------------------- helpers -------------------------

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "build",
    "cmake-build-debug",
    "cmake-build-release",
    "__pycache__",
}


def is_in_project(path: Optional[str], project_root: str) -> bool:
    if not path:
        return False
    pr = os.path.abspath(project_root)
    # IMPORTANT: if libclang gave us a relative path (e.g., "src/hash.c"),
    # treat it as relative to the project root, not the Python CWD.
    p = path if os.path.isabs(path) else os.path.abspath(os.path.join(pr, path))
    try:
        rel = os.path.relpath(p, pr)
        return not rel.startswith("..")
    except Exception:
        return False


def module_path(project_root: str, file_path: str) -> str:
    rel = os.path.relpath(os.path.abspath(file_path), os.path.abspath(project_root))
    stem = os.path.splitext(rel)[0]
    return stem.replace(os.sep, ".")


def gather_sources(root: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for dp, dirs, files in os.walk(root):
        # prune
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for fn in files:
            if fn.endswith(exts):
                out.append(os.path.join(dp, fn))
    return out


def get_source_snippet(cur: Cursor, line_limit: int = 400) -> str:
    try:
        ext = cur.extent
        if not ext.start.file or not ext.end.file:
            return ""
        path = ext.start.file.name
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        s = max(1, ext.start.line)
        e = min(len(lines), ext.end.line)
        snippet = "".join(lines[s - 1 : e])
        # avoid megabyte nodes
        if snippet.count("\n") > line_limit:
            snippet = "\n".join(snippet.splitlines()[:line_limit]) + "\n# [truncated]\n"
        return snippet
    except Exception:
        return ""


def peel_type_layers(ty):
    """
    Remove pointer/array/elaborated layers to reach the meaningful core type.
    Returns the terminal type (may be TYPEDEF or RECORD, etc.)
    """
    if ty is None:
        return None
    prev = None
    cur = ty
    while True:
        prev = cur
        if cur.kind in (
            TypeKind.POINTER,
            TypeKind.LVALUEREFERENCE,
            TypeKind.RVALUEREFERENCE,
        ):
            cur = cur.get_pointee()
        elif cur.kind in (
            TypeKind.CONSTANTARRAY,
            TypeKind.INCOMPLETEARRAY,
            TypeKind.VARIABLEARRAY,
        ):
            cur = cur.element_type
        elif cur.kind == TypeKind.ELABORATED:
            cur = cur.get_named_type()
        else:
            break
    return cur or prev


def collect_type_deps(ty) -> Tuple[Optional[Cursor], Optional[Cursor]]:
    """
    Return (typedef_decl, record_decl) cursors if present in the (possibly layered) type.
    typedef_decl is returned when the immediate type is a typedef.
    record_decl is returned for underlying struct/union (canonical).
    """
    if ty is None:
        return (None, None)

    # Immediate typedef alias (if present)
    typedef_decl = None
    base = peel_type_layers(ty)
    if base and base.kind == TypeKind.TYPEDEF:
        typedef_decl = base.get_declaration()

    # Canonical to reach underlying record (struct/union)
    rec_decl = None
    can = ty.get_canonical()
    can_base = peel_type_layers(can)
    if can_base and can_base.kind in (TypeKind.RECORD,):
        rec_decl = can_base.get_declaration()

    return (typedef_decl, rec_decl)


def want_kind(kind: str, include: Set[str]) -> bool:
    return kind in include


# ------------------------- compile commands DB -------------------------


def _find_compdb_dir(start: str) -> str | None:
    d = os.path.abspath(start)
    while True:
        if any(
            os.path.isfile(os.path.join(d, f))
            for f in ("compile_commands.json", "compile_flags.txt")
        ):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


def get_compile_db(root: str):
    """Only try to open a compilation DB if one actually exists, to avoid libclang's stderr spam."""
    db_dir = _find_compdb_dir(root)
    if not db_dir:
        return None
    try:
        return cindex.CompilationDatabase.fromDirectory(db_dir)
    except cindex.CompilationDatabaseError:
        return None


def _sanitize_compile_args(
    argv: list[str], file_abs: str | None = None, base_dir: str | None = None
) -> list[str]:
    """
    Strip compile-only flags, output flags, and the *input filename* present in argv.
    `file_abs` must be absolute; `base_dir` is the compile command's working dir.
    """
    import os

    out = []
    i = 0
    drop = {
        "-c",
        "-S",
        "-E",
        "-M",
        "-MM",
        "-MD",
        "-MMD",
        "-MP",
        "-MG",
        "-fcolor-diagnostics",
    }
    while i < len(argv):
        a = argv[i]

        # drop compile-only / depgen flags
        if a in drop:
            i += 1
            continue
        if a in ("-o", "-MF", "-MT", "-MQ"):
            i += 2
            continue
        if a.startswith(("-o", "-MF", "-MT", "-MQ")):
            i += 1
            continue

        # drop bare input filename token
        if file_abs and not a.startswith("-"):
            # resolve token against the compile command's directory
            try:
                cand_abs = (
                    a
                    if os.path.isabs(a)
                    else os.path.abspath(os.path.join(base_dir or "", a))
                )
            except Exception:
                cand_abs = a
            if os.path.abspath(cand_abs) == os.path.abspath(file_abs):
                i += 1
                continue

        out.append(a)
        i += 1
    return out


def _abspath_includes(args: list[str], base_dir: str) -> list[str]:
    import os

    out = []
    i = 0

    def absify(p):
        if not p or p.startswith("/"):
            return p
        return os.path.abspath(os.path.join(base_dir, p))

    while i < len(args):
        a = args[i]
        if a in ("-I", "-isystem", "-iquote"):
            if i + 1 < len(args):
                out.extend([a, absify(args[i + 1])])
                i += 2
                continue
        if a.startswith("-I") and len(a) > 2:
            out.append("-I" + absify(a[2:]))
            i += 1
            continue
        if a.startswith("-isystem") and len(a) > 8:
            out.append("-isystem" + absify(a[8:]))
            i += 1
            continue
        if a.startswith("-iquote") and len(a) > 7:
            out.append("-iquote" + absify(a[7:]))
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def args_for_file(db, default_args: list[str], filepath: str):
    """
    Use compile_commands.json if present:
      - set -working-directory=<dir> (single token)
      - absolutize include paths against entry.directory
      - strip compile-/output-only flags and the input filename token
      - merge your defaults (lang/std/sys-includes)
    Returns (parse_path, final_args) for clang.
    """
    base_args = list(default_args)
    if not db:
        return filepath, base_args

    cmds = db.getCompileCommands(filepath)
    if not cmds:
        return filepath, base_args

    cc = cmds[0]
    argv = list(cc.arguments)[1:]  # drop compiler
    file_abs = os.path.abspath(filepath)
    argv = _sanitize_compile_args(argv, file_abs=file_abs, base_dir=cc.directory)
    argv = _abspath_includes(argv, cc.directory or os.path.dirname(filepath))

    final = []
    if cc.directory:
        final.append(f"-working-directory={cc.directory}")  # ONE token

    final += argv

    # keep defaults (duplicates harmless)
    for a in base_args:
        if a not in final:
            final.append(a)

    # Prefer a path relative to working dir if we have one
    if cc.directory and os.path.isabs(filepath):
        try:
            parse_path = os.path.relpath(filepath, cc.directory)
        except Exception:
            parse_path = filepath
    else:
        parse_path = filepath

    return parse_path, final


# ------------------------- Core graph builder -------------------------


class NodeInfo:
    __slots__ = (
        "id",
        "kind",
        "code",
        "file",
        "usr",
        "cursor",
        "start_line",
        "end_line",
        "code_sha256",
        "display",
    )

    def __init__(
        self,
        id,
        kind,
        code,
        file,
        usr,
        cursor,
        start_line,
        end_line,
        code_sha256,
        display,
    ):
        self.id = id
        self.kind = kind
        self.code = code
        self.file = file
        self.usr = usr
        self.cursor = cursor
        self.start_line = start_line
        self.end_line = end_line
        self.code_sha256 = code_sha256
        self.display = display


class CodeGraphBuilder:
    def __init__(
        self,
        project_root: str,
        include_kinds: Set[str],
        extra_args: List[str],
        prefer_headers: bool,
        trace_args: bool = False,
    ):
        self.root = os.path.abspath(project_root)
        self.include = include_kinds
        self.extra_args = extra_args
        self.prefer_headers = prefer_headers
        self.trace_args = trace_args

        self.index = cindex.Index.create()
        self.db = get_compile_db(self.root)

        # USR -> NodeInfo
        self.nodes: Dict[str, NodeInfo] = {}
        # Quick maps for edge resolution
        self.func_usrs: Set[str] = set()
        self.struct_usrs: Set[str] = set()
        self.global_usrs: Set[str] = set()
        self.typedef_usrs: Set[str] = set()
        self.alias_of: Dict[str, str] = {}  # record_usr -> typedef_usr (merge map)

        self.G = nx.Graph()

    # ---------- node creation ----------

    def _merge_records_into_typedefs(self):
        """
        Prefer typedef nodes over their underlying record (struct/union).
        Build alias_of[record_usr] = typedef_usr, then drop the record node.
        Order-independent and robust even if multiple typedefs name the same record.
        """
        # record_usr -> NodeInfo
        records = {usr: ni for usr, ni in self.nodes.items() if ni.kind == "struct"}

        # For every typedef, see if it aliases a record we collected
        rec_to_tdefs = {}  # rec_usr -> [typedef NodeInfo]
        for tusr, tni in list(self.nodes.items()):
            if tni.kind != "typedef":
                continue
            tcur = tni.cursor
            try:
                _, rec = collect_type_deps(tcur.underlying_typedef_type)
            except Exception:
                rec = None
            if not rec:
                continue
            if (
                not rec.location
                or not rec.location.file
                or not is_in_project(rec.location.file.name, self.root)
            ):
                continue
            rec_usr = rec.get_usr()
            if rec_usr in records:
                rec_to_tdefs.setdefault(rec_usr, []).append(tni)

        # Choose a preferred typedef per record
        def tdef_sort_key(ni):
            loc = ni.cursor.location
            pos = (
                getattr(loc.file, "name", "") if (loc and loc.file) else "",
                getattr(loc, "line", 0),
                getattr(loc, "column", 0),
            )
            return (pos, ni.cursor.spelling or "~")

        for rec_usr, tdefs in rec_to_tdefs.items():
            preferred = sorted(tdefs, key=tdef_sort_key)[0]
            self.alias_of[rec_usr] = preferred.usr

        # Drop record nodes aliased by a typedef, and update struct set
        for rec_usr, tdef_usr in list(self.alias_of.items()):
            if rec_usr in self.nodes:
                del self.nodes[rec_usr]
            if rec_usr in self.struct_usrs:
                self.struct_usrs.remove(rec_usr)

    def _add_node(self, cur: Cursor, kind: str):
        if not want_kind(kind, self.include):
            return

        usr = cur.get_usr() or ""
        if not usr:
            return

        loc = cur.location
        if not loc or not loc.file:
            return

        # libclang may return a RELATIVE path (e.g., "src/hash.c") when using a compile DB.
        raw_path = loc.file.name
        abs_path = (
            raw_path
            if os.path.isabs(raw_path)
            else os.path.abspath(os.path.join(self.root, raw_path))
        )
        if not is_in_project(abs_path, self.root):
            return
        rel_path = os.path.relpath(abs_path, self.root)

        name = cur.spelling or ""
        if not name and kind == "struct":
            # anonymous struct without a typedef — skip as a node
            return

        # Build display/key from the ABS path so module inference is correct
        mod = module_path(self.root, abs_path)
        display = f"{mod}.{name}" if name else mod
        node_id = f"{display}::{kind}"

        # Read code from the ABS path (don’t rely on current working dir)
        snippet = ""
        ext = cur.extent
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            s = max(1, ext.start.line) if ext and ext.start else 1
            e = min(len(lines), ext.end.line) if ext and ext.end else s
            snippet = "".join(lines[s - 1 : e])
        except Exception:
            snippet = ""

        start_line = ext.start.line if ext and ext.start else None
        end_line = ext.end.line if ext and ext.end else None

        code_sha = hashlib.sha256(
            (snippet or "").encode("utf-8", errors="ignore")
        ).hexdigest()

        if usr in self.nodes:
            return

        ni = NodeInfo(
            id=node_id,
            kind=kind,
            code=snippet,
            file=rel_path,  # stored RELATIVE to project root
            usr=usr,
            cursor=cur,
            start_line=start_line,
            end_line=end_line,
            code_sha256=code_sha,
            display=display,
        )
        self.nodes[usr] = ni

        if kind == "function":
            self.func_usrs.add(usr)
        elif kind == "struct":
            self.struct_usrs.add(usr)
        elif kind == "global":
            self.global_usrs.add(usr)
        elif kind == "typedef":
            self.typedef_usrs.add(usr)

    def _collect_defs_in_tu(self, tu):
        for cur in tu.cursor.walk_preorder():
            if not cur.location or not cur.location.file:
                continue
            if not is_in_project(cur.location.file.name, self.root):
                continue

            # functions (definitions only)
            if cur.kind == cindex.CursorKind.FUNCTION_DECL and cur.is_definition():
                self._add_node(cur, "function")

            # file-scope globals (definitions only)
            elif cur.kind == cindex.CursorKind.VAR_DECL and cur.is_definition():
                if (
                    cur.semantic_parent
                    and cur.semantic_parent.kind == cindex.CursorKind.TRANSLATION_UNIT
                ):
                    self._add_node(cur, "global")

            # struct/union *definitions* (keep the check)
            elif (
                cur.kind
                in (cindex.CursorKind.STRUCT_DECL, cindex.CursorKind.UNION_DECL)
                and cur.is_definition()
            ):
                self._add_node(cur, "struct")

            elif cur.kind == cindex.CursorKind.TYPEDEF_DECL and cur.is_definition():
                self._add_node(cur, "typedef")

    def parse_all(self):
        if self.db:
            # With a compile DB: only parse .c files that actually have commands
            c_sources = gather_sources(self.root, (".c", ".C"))
            files = [
                f for f in c_sources if self.db.getCompileCommands(os.path.abspath(f))
            ]
            # Safety fallback: if the DB didn’t match anything, just use the .c list
            if not files:
                files = c_sources
        else:
            # No DB: optionally include headers, if you want to scan them directly
            exts = (".c", ".C") + (
                (".h", ".hpp", ".hh", ".hxx") if self.prefer_headers else ()
            )
            files = gather_sources(self.root, exts)

        if not files:
            raise SystemExit(f"No source files found under {self.root}")

        for src in files:
            parse_path, args = args_for_file(self.db, self.extra_args, src)
            if self.trace_args:
                print("[trace]", parse_path, "args=", args)
            try:
                tu = self.index.parse(parse_path, args=args)
            except cindex.TranslationUnitLoadError as e:
                print(f"[warn] failed to parse {src}: {e}")
                continue

            # record diagnostics but keep going
            for d in tu.diagnostics:
                if d.severity >= d.Warning:
                    print(
                        f"[diag] {src}:{d.location.line}:{d.location.column} {d.spelling}",
                        file=sys.stderr,
                    )

            self._collect_defs_in_tu(tu)

        # sync nodes into graph
        self._merge_records_into_typedefs()
        self.G.graph.update(
            {
                "undirected": True,
                "kinds_included": ",".join(sorted(self.include)),
                "project_root": self.root,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "tool": "cgraph",
                "version": "1.0.0",
            }
        )
        for ni in self.nodes.values():
            self.G.add_node(
                ni.id,
                display=ni.display,
                kind=ni.kind,
                code=ni.code,
                file=ni.file,
                usr=ni.usr,
                start_line=ni.start_line,
                end_line=ni.end_line,
                code_sha256=ni.code_sha256,
            )

    # ---------- edge creation ----------

    def _norm(self, usr: str) -> str:
        return self.alias_of.get(usr, usr)

    def _maybe_add_edge_by_usrs(self, from_usr: str, to_usr: str):
        from_usr = self._norm(from_usr)
        to_usr = self._norm(to_usr)
        if from_usr == to_usr:
            return
        a = self.nodes.get(from_usr)
        b = self.nodes.get(to_usr)
        if not a or not b:
            return
        if a.id == b.id:
            return
        self.G.add_edge(a.id, b.id)

    def _add_func_edges(self, fcur: Cursor, fusr: str):
        # 1) calls and references to functions
        for n in fcur.walk_preorder():
            # Function used via name (call or address-of): DECL_REF_EXPR -> referenced FUNCTION_DECL
            if n.kind == CursorKind.DECL_REF_EXPR and n.referenced:
                ref = n.referenced
                if ref.kind == CursorKind.FUNCTION_DECL:
                    rusr = ref.get_usr()
                    if rusr in self.func_usrs and rusr in self.nodes:
                        # ensure referenced symbol is in-project (by definition location)
                        if is_in_project(
                            (
                                ref.location.file.name
                                if ref.location and ref.location.file
                                else None
                            ),
                            self.root,
                        ):
                            self._maybe_add_edge_by_usrs(fusr, rusr)

            # Global var references: DECL_REF_EXPR -> VAR_DECL
            if n.kind == CursorKind.DECL_REF_EXPR and n.referenced:
                ref = n.referenced
                if ref.kind == CursorKind.VAR_DECL:
                    rusr = ref.get_usr()
                    if rusr in self.global_usrs:
                        if is_in_project(
                            (
                                ref.location.file.name
                                if ref.location and ref.location.file
                                else None
                            ),
                            self.root,
                        ):
                            self._maybe_add_edge_by_usrs(fusr, rusr)

            # Member references -> FIELD_DECL; parent is the record (struct/union)
            if n.kind in (CursorKind.MEMBER_REF_EXPR,):
                ref = n.referenced
                if ref and ref.kind == CursorKind.FIELD_DECL:
                    parent = ref.semantic_parent
                    if parent and parent.kind in (
                        CursorKind.STRUCT_DECL,
                        CursorKind.UNION_DECL,
                    ):
                        rusr = parent.get_usr()
                        if rusr in self.struct_usrs:
                            if is_in_project(
                                (
                                    parent.location.file.name
                                    if parent.location and parent.location.file
                                    else None
                                ),
                                self.root,
                            ):
                                self._maybe_add_edge_by_usrs(fusr, rusr)

        # 2) types in return / params / local vars (typedef + underlying record)
        def visit_type(ty):
            tdef, rec = collect_type_deps(ty)
            if tdef:
                rusr = tdef.get_usr()
                if rusr in self.typedef_usrs:
                    if is_in_project(
                        (
                            tdef.location.file.name
                            if tdef.location and tdef.location.file
                            else None
                        ),
                        self.root,
                    ):
                        self._maybe_add_edge_by_usrs(fusr, rusr)
            if rec:
                rusr = rec.get_usr()
                if rusr in self.struct_usrs:
                    if is_in_project(
                        (
                            rec.location.file.name
                            if rec.location and rec.location.file
                            else None
                        ),
                        self.root,
                    ):
                        self._maybe_add_edge_by_usrs(fusr, rusr)

        # return type
        visit_type(fcur.result_type)
        # params
        for ch in fcur.get_children():
            if ch.kind == CursorKind.PARM_DECL:
                visit_type(ch.type)
        # locals
        for n in fcur.walk_preorder():
            if n.kind == CursorKind.VAR_DECL:
                visit_type(n.type)

    def _add_struct_edges(self, scur: Cursor, susr: str):
        # From struct fields to their types (typedefs + other structs)
        for ch in scur.get_children():
            if ch.kind == CursorKind.FIELD_DECL:
                tdef, rec = collect_type_deps(ch.type)
                if tdef:
                    rusr = tdef.get_usr()
                    if rusr in self.typedef_usrs:
                        if is_in_project(
                            (
                                tdef.location.file.name
                                if tdef.location and tdef.location.file
                                else None
                            ),
                            self.root,
                        ):
                            self._maybe_add_edge_by_usrs(susr, rusr)
                if rec:
                    rusr = rec.get_usr()
                    if rusr in self.struct_usrs:
                        if is_in_project(
                            (
                                rec.location.file.name
                                if rec.location and rec.location.file
                                else None
                            ),
                            self.root,
                        ):
                            self._maybe_add_edge_by_usrs(susr, rusr)

    def _add_typedef_edges(self, tcur: Cursor, tusr: str):
        base = peel_type_layers(tcur.underlying_typedef_type)
        if base and base.kind == TypeKind.TYPEDEF:
            t2 = base.get_declaration()
            if t2:
                rusr = self._norm(t2.get_usr())
                if rusr in self.typedef_usrs and rusr != tusr:
                    if is_in_project(
                        (
                            t2.location.file.name
                            if t2.location and t2.location.file
                            else None
                        ),
                        self.root,
                    ):
                        self._maybe_add_edge_by_usrs(tusr, rusr)

        # Underlying record
        _, rec = collect_type_deps(tcur.underlying_typedef_type)
        if rec:
            rec_usr = self._norm(rec.get_usr())
            # If the record aliases to this typedef, do NOT add an edge (would self-loop)
            if rec_usr == tusr:
                return
            if rec_usr in self.struct_usrs:
                if is_in_project(
                    (
                        rec.location.file.name
                        if rec.location and rec.location.file
                        else None
                    ),
                    self.root,
                ):
                    self._maybe_add_edge_by_usrs(tusr, rec_usr)

    def add_edges(self):
        # Build a per-file parse again only for cursors we already have (so we can walk bodies)
        # We can reuse the earlier parse loop by grouping by file and reparsing.
        if self.db:
            files = [
                f
                for f in gather_sources(self.root, (".c", ".C"))
                if self.db.getCompileCommands(os.path.abspath(f))
            ]
            if not files:
                files = gather_sources(self.root, (".c", ".C"))
        else:
            files = sorted(
                {
                    (f if os.path.isabs(f) else os.path.join(self.root, f))
                    for f in (ni.file for ni in self.nodes.values())
                }
            )

        for src in files:
            abs_src = src if os.path.isabs(src) else os.path.join(self.root, src)
            parse_path, args = args_for_file(self.db, self.extra_args, abs_src)
            if self.trace_args:
                print("[trace]", parse_path, "args=", args)
            try:
                tu = self.index.parse(parse_path, args=args)
            except cindex.TranslationUnitLoadError as e:
                print(f"[warn] failed to reparse {src}: {e}")
                continue

            # walk and hook edges
            for cur in tu.cursor.get_children():
                if not cur.location or not cur.location.file:
                    continue
                if not is_in_project(cur.location.file.name, self.root):
                    continue

                if cur.kind == CursorKind.FUNCTION_DECL and cur.is_definition():
                    fusr = cur.get_usr()
                    if fusr in self.func_usrs:
                        self._add_func_edges(cur, fusr)

                elif (
                    cur.kind in (CursorKind.STRUCT_DECL, CursorKind.UNION_DECL)
                    and cur.is_definition()
                ):
                    susr = cur.get_usr()
                    if susr in self.struct_usrs:
                        self._add_struct_edges(cur, susr)

                elif cur.kind == CursorKind.TYPEDEF_DECL and cur.is_definition():
                    tusr = cur.get_usr()
                    if tusr in self.typedef_usrs:
                        self._add_typedef_edges(cur, tusr)

    # ---------- export ----------

    def export_graphml(self, outpath: str):
        nx.write_graphml(self.G, outpath)


# ------------------------- Other outputs -------------------------


def _label_for(node_id: str, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "full":
        return node_id
    # short
    return node_id.split(".")[-1]


def save_png(
    G, path: str, label_mode: str = "short", layout: str = "spring", dpi: int = 180
):
    if len(G) == 0:
        plt.figure()
        plt.text(0.5, 0.5, "(empty graph)", ha="center")
        plt.axis("off")
        plt.savefig(path, dpi=dpi)
        plt.close()
        return

    # Layouts
    if layout == "kk":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "fr":
        pos = nx.fruchterman_reingold_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)

    color_map = {
        "function": "#8fd19e",  # light green
        "struct": "#87ceeb",  # sky blue
        "global": "#f7b267",  # orange
        "typedef": "#b39ddb",  # lavender
    }
    node_colors = [color_map.get(G.nodes[n].get("kind"), "#cccccc") for n in G.nodes()]
    node_sizes = [
        300 + 40 * G.degree(n) for n in G.nodes()
    ]  # degree = rough importance

    # Figure size scales mildly with graph size (don’t go insane)
    w = min(22, max(8, 0.18 * len(G)))
    h = min(16, max(6, 0.12 * len(G)))
    plt.figure(figsize=(w, h))
    nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.4)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.5,
        edgecolors="black",
    )

    if label_mode != "none":
        labels = {
            n: _label_for(data.get("display", n), label_mode)
            for n, data in G.nodes(data=True)
        }
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    # Legend (only for kinds that actually appear)
    present = {G.nodes[n].get("kind") for n in G.nodes()}
    handles = [
        Patch(color=color_map[k], label=k)
        for k in ("function", "struct", "global", "typedef")
        if k in present
    ]
    if handles:
        plt.legend(handles=handles, loc="best", fontsize=8, frameon=False)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def export_html_pyvis(G, path: str, label_mode: str = "short", physics: bool = True):
    net = Network(
        height="100%",
        width="100%",
        directed=False,
        notebook=False,
        cdn_resources="in_line",
    )
    net.toggle_physics(physics)

    color_map = {
        "function": "#8fd19e",
        "struct": "#87ceeb",
        "global": "#f7b267",
        "typedef": "#b39ddb",
    }

    for n, data in G.nodes(data=True):
        human = data.get("display", n)
        label = _label_for(human, label_mode)
        # Keep tooltip reasonable
        code = (data.get("code") or "").strip()
        if len(code) > 2000:
            code = code[:2000] + "\n# [truncated]"
        title = f"<b>{html.escape(human)}</b><br><i>{html.escape(data.get('kind',''))}</i><br><pre>{html.escape(code)}</pre>"
        net.add_node(
            n,
            label=label,
            title=title,
            color=color_map.get(data.get("kind"), "#cccccc"),
            shape="dot",
            value=max(1, G.degree(n)),  # size by degree
        )

    for u, v in G.edges():
        net.add_edge(u, v)

    # Writes full HTML (self-contained)
    net.show(path)


def export_html_visjs(G, path: str, label_mode: str = "short", physics: bool = True):
    def _label_for(node_id: str, mode: str) -> str:
        if mode == "none":
            return ""
        if mode == "full":
            return node_id
        return node_id.split(".")[-1]

    color_map = {
        "function": "#8fd19e",
        "struct": "#87ceeb",
        "global": "#f7b267",
        "typedef": "#b39ddb",
    }

    nodes = []
    module_counts = {}
    for n, data in G.nodes(data=True):
        human = data.get("display", n)
        label = _label_for(human, label_mode)
        # module prefix = everything before the last dot
        if "." in human:
            module = human.rsplit(".", 1)[0]
        else:
            module = ""  # root
        module_counts[module] = module_counts.get(module, 0) + 1

        code = (data.get("code") or "").strip()
        if len(code) > 2000:
            code = code[:2000] + "\n# [truncated]"
        title = (
            f"<b>{html.escape(human)}</b><br>"
            f"<i>{html.escape(data.get('kind',''))}</i>"
            f"<br><pre style='max-width:800px; white-space:pre-wrap'>{html.escape(code)}</pre>"
        )
        nodes.append(
            {
                "id": n,
                "display": human,  # for search
                "module": module,  # for module filtering
                "label": label,
                "title": title,
                "kind": data.get("kind", ""),
                "color": color_map.get(data.get("kind"), "#ccc"),
                "value": max(1, G.degree(n)),  # size by degree
            }
        )

    # Build a sorted module list (by frequency desc, then name)
    modules = [{"name": m, "count": c} for m, c in module_counts.items()]
    modules.sort(key=lambda x: (-x["count"], x["name"] or "~"))

    edges = [{"from": u, "to": v} for u, v in G.edges()]

    html_str = f"""<!doctype html>
        <html>
        <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>cgraph</title>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
        :root {{ --ui-font: 14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
        html, body {{ height: 100%; margin: 0; }}
        #controls {{
            padding: 8px; font: var(--ui-font); display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: center;
        }}
        #left, #right {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
        .chip {{ display:inline-block; padding:2px 8px; border-radius:12px; color:#222; }}
        .grp {{ display:flex; align-items:center; gap:8px; }}
        #mynetwork {{ height: calc(100vh - 64px); border-top: 1px solid #ddd; }}
        input[type="text"] {{ padding: 6px 8px; font: var(--ui-font); width: 280px; }}
        select {{ padding: 6px 8px; font: var(--ui-font); max-width: 360px; }}
        button {{ padding: 6px 10px; font: var(--ui-font); cursor: pointer; }}
        .count {{ opacity: .7; }}
        </style>
        </head>
        <body>
        <div id="controls">
        <div id="left">
            <label class="grp"><input id="physicsToggle" type="checkbox" {"checked" if physics else ""}/> physics</label>
            <span class="grp">
            <label><input type="checkbox" class="kindChk" data-kind="function" checked/> <span class="chip" style="background:#8fd19e">function</span></label>
            <label><input type="checkbox" class="kindChk" data-kind="struct"   checked/> <span class="chip" style="background:#87ceeb">struct</span></label>
            <label><input type="checkbox" class="kindChk" data-kind="global"   checked/> <span class="chip" style="background:#f7b267">global</span></label>
            <label><input type="checkbox" class="kindChk" data-kind="typedef"  checked/> <span class="chip" style="background:#b39ddb">typedef</span></label>
            </span>
            <span class="grp">
            <select id="moduleSelect" title="Filter by module prefix">
                <option value="">All modules</option>
            </select>
            </span>
        </div>
        <div id="right">
            <input id="searchBox" type="text" placeholder="Search nodes (id/display/label)…" />
            <label class="grp"><input id="includeNeighbors" type="checkbox" checked/> include neighbors</label>
            <button id="fitBtn">Fit</button>
            <button id="resetBtn">Reset</button>
            <span class="count" id="counts"></span>
        </div>
        </div>
        <div id="mynetwork"></div>

        <script>
        // Raw data from Python
        const ALL_NODES = {json.dumps(nodes)};
        const ALL_EDGES = {json.dumps(edges)};
        const MODULES   = {json.dumps(modules)};

        // Build DataSets
        const allNodes = new vis.DataSet(ALL_NODES);
        const allEdges = new vis.DataSet(ALL_EDGES);

        // Precompute adjacency for neighbor expansion
        const neighbors = new Map();
        for (const e of ALL_EDGES) {{
            if (!neighbors.has(e.from)) neighbors.set(e.from, new Set());
            if (!neighbors.has(e.to))   neighbors.set(e.to,   new Set());
            neighbors.get(e.from).add(e.to);
            neighbors.get(e.to).add(e.from);
        }}

        // Populate module dropdown
        const moduleSelect = document.getElementById('moduleSelect');
        MODULES.forEach(m => {{
            const label = (m.name || "(root)") + " (" + m.count + ")";
            const opt = new Option(label, m.name);
            moduleSelect.add(opt);
        }});

        // UI state
        const state = {{
            physics: {str(physics).lower()},
            enabledKinds: new Set(["function","struct","global","typedef"]),
            modulePrefix: "",   // empty = all
            search: "",
            includeNeighbors: true,
            visible: new Set(),  // computed
        }};

        // Helpers
        const container = document.getElementById('mynetwork');
        const countsEl  = document.getElementById('counts');

        function nodeMatchesSearch(node, q) {{
            if (!q) return true;
            const s = q.toLowerCase();
            return (node.display && node.display.toLowerCase().includes(s)) ||
                (node.id && String(node.id).toLowerCase().includes(s)) ||
                (node.label && String(node.label).toLowerCase().includes(s));
        }}

        function nodeMatchesModule(node, prefix) {{
            if (!prefix) return true;
            // node.module is the module path (file-based), e.g., "src.foo.bar"
            return node.module === prefix || node.module.startsWith(prefix + ".");
        }}

        // DataViews with dynamic filters (closures read 'state.visible')
        const nodeView = new vis.DataView(allNodes, {{
            filter: function (item) {{ return state.visible.has(item.id); }}
        }});
        const edgeView = new vis.DataView(allEdges, {{
            filter: function (item) {{ return state.visible.has(item.from) && state.visible.has(item.to); }}
        }});

        const options = {{
            physics: {{ enabled: state.physics }},
            interaction: {{ hover: true, tooltipDelay: 120 }},
            nodes: {{ shape: 'dot', borderWidth: 1 }},
            edges: {{ smooth: true, opacity: 0.6 }}
        }};

        const network = new vis.Network(container, {{ nodes: nodeView, edges: edgeView }}, options);

        // Filter application
        function applyFilters() {{
            const q = state.search.trim().toLowerCase();

            // Step 1: nodes that pass kind + module + text
            const matched = new Set();
            allNodes.forEach((n) => {{
            if (!state.enabledKinds.has(n.kind)) return;
            if (!nodeMatchesModule(n, state.modulePrefix)) return;
            if (nodeMatchesSearch(n, q)) matched.add(n.id);
            }});

            // Step 2: optionally include neighbors (still honor kind+module)
            const visible = new Set(matched);
            if (state.includeNeighbors && q) {{
            for (const id of matched) {{
                const nbrs = neighbors.get(id);
                if (!nbrs) continue;
                nbrs.forEach(nid => {{
                const node = allNodes.get(nid);
                if (node && state.enabledKinds.has(node.kind) && nodeMatchesModule(node, state.modulePrefix)) {{
                    visible.add(nid);
                }}
                }});
            }}
            }}

            // If no search term, show all of the enabled kinds within module
            if (!q) {{
            allNodes.forEach((n) => {{
                if (state.enabledKinds.has(n.kind) && nodeMatchesModule(n, state.modulePrefix)) visible.add(n.id);
            }});
            }}

            state.visible = visible;

            nodeView.refresh();
            edgeView.refresh();

            // Update counts
            const total = allNodes.length;
            const visCount = state.visible.size;
            countsEl.textContent = visCount + " / " + total + " nodes";
        }}

        // Debounce for search box
        function debounce(fn, ms) {{ let t; return function(...args) {{ clearTimeout(t); t = setTimeout(() => fn.apply(this,args), ms); }} }}

        // Wire UI
        document.getElementById('physicsToggle').addEventListener('change', (e) => {{
            state.physics = e.target.checked;
            network.setOptions({{ physics: {{ enabled: state.physics }} }});
        }});

        document.querySelectorAll('.kindChk').forEach(cb => {{
            cb.addEventListener('change', (e) => {{
            const k = e.target.getAttribute('data-kind');
            if (e.target.checked) state.enabledKinds.add(k); else state.enabledKinds.delete(k);
            applyFilters();
            }});
        }});

        document.getElementById('includeNeighbors').addEventListener('change', (e) => {{
            state.includeNeighbors = e.target.checked;
            applyFilters();
        }});

        document.getElementById('searchBox').addEventListener('input', debounce(() => {{
            state.search = document.getElementById('searchBox').value || "";
            applyFilters();
        }}, 150));

        document.getElementById('fitBtn').addEventListener('click', () => {{ network.fit({{ animation: true }}); }});

        document.getElementById('resetBtn').addEventListener('click', () => {{
            // reset UI
            document.getElementById('searchBox').value = "";
            state.search = "";
            state.includeNeighbors = true;
            document.getElementById('includeNeighbors').checked = true;
            state.enabledKinds = new Set(["function","struct","global","typedef"]);
            document.querySelectorAll('.kindChk').forEach(cb => cb.checked = true);
            // reset module dropdown
            state.modulePrefix = "";
            moduleSelect.value = "";
            applyFilters();
            network.fit({{ animation: true }});
        }});

        moduleSelect.addEventListener('change', (e) => {{
            state.modulePrefix = e.target.value || "";
            applyFilters();
        }});

        // Initial render
        applyFilters();
        </script>
        </body>
        </html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)


def _maybe_trim_code(s: str | None, omit: bool, limit: int | None) -> str | None:
    if omit:
        return None
    if s is None:
        return None
    if limit is not None and len(s) > limit:
        return s[:limit] + "\n# [truncated]"
    return s


def export_json_full(G, path, omit_code: bool, code_limit: int | None):
    data = {
        "meta": G.graph,
        "nodes": [],
        "edges": [],
    }
    for n, d in G.nodes(data=True):
        data["nodes"].append(
            {
                "id": n,
                "display": d.get("display", n),
                "kind": d.get("kind"),
                "file": d.get("file"),
                "usr": d.get("usr"),
                "start_line": d.get("start_line"),
                "end_line": d.get("end_line"),
                "code_sha256": d.get("code_sha256"),
                "code": _maybe_trim_code(d.get("code"), omit_code, code_limit),
                "degree": G.degree(n),
            }
        )
    # undirected: networkx Graph ensures unique pairs
    for u, v in G.edges():
        data["edges"].append({"source": u, "target": v})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_json_compact(G, path, omit_code: bool, code_limit: int | None):
    # map node ids to dense indices
    ids = list(G.nodes())
    index = {nid: i for i, nid in enumerate(ids)}
    nodes = []
    for nid in ids:
        d = G.nodes[nid]
        nodes.append(
            {
                "i": index[nid],
                "id": nid,
                "display": d.get("display", nid),
                "kind": d.get("kind"),
                "file": d.get("file"),
                "usr": d.get("usr"),
                "start_line": d.get("start_line"),
                "end_line": d.get("end_line"),
                "code_sha256": d.get("code_sha256"),
                "code": _maybe_trim_code(d.get("code"), omit_code, code_limit),
            }
        )
    edges = [[index[u], index[v]] for u, v in G.edges()]
    data = {
        "meta": G.graph,
        "index_map": {nid: i for nid, i in index.items()},  # useful for joins
        "nodes": nodes,
        "edges": edges,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_json_ndjson(G, path, omit_code: bool, code_limit: int | None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", **G.graph}) + "\n")
        for n, d in G.nodes(data=True):
            rec = {
                "type": "node",
                "id": n,
                "display": d.get("display", n),
                "kind": d.get("kind"),
                "file": d.get("file"),
                "usr": d.get("usr"),
                "start_line": d.get("start_line"),
                "end_line": d.get("end_line"),
                "code_sha256": d.get("code_sha256"),
                "code": _maybe_trim_code(d.get("code"), omit_code, code_limit),
                "degree": G.degree(n),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for u, v in G.edges():
            f.write(json.dumps({"type": "edge", "source": u, "target": v}) + "\n")


# ------------------------- CLI -------------------------


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build a C dependency graph (undirected) from in-project definitions."
    )
    ap.add_argument("project_root", help="Path to project root.")
    ap.add_argument(
        "-o", "--output", default="cgraph.graphml", help="GraphML output path."
    )
    ap.add_argument(
        "--include",
        default="function,struct,typedef,global",
        help="Comma list of node kinds: function,struct,typedef,global",
    )
    ap.add_argument(
        "--no-headers",
        action="store_true",
        help="Do not scan header files (.h/.hpp) directly.",
    )
    ap.add_argument(
        "--extra-arg",
        dest="extra_args",
        action="append",
        default=[],
        help="Extra compiler arg (repeatable). Example: --extra-arg=-Iinclude"
        "\nNote: if you have a Makefile, you can first use it to create a compile_commands.json (e.g., with `bear -- make -j`) in the project's root, and then you won't need to specify the extra includes manually.",
    )
    ap.add_argument("--std", default="c11", help="C standard (default c11).")
    ap.add_argument(
        "--sys-include",
        action="append",
        default=[],
        help="Add a system include dir (repeatable).",
    )
    ap.add_argument(
        "--trace-args",
        action="store_true",
        help="Print final libclang args per file (for debugging).",
    )
    ap.add_argument(
        "--png",
        action="store_true",
        help="Write a static PNG visualization.",
    )
    ap.add_argument(
        "--html",
        action="store_true",
        help="Write an interactive HTML visualization.",
    )
    ap.add_argument(
        "--html-engine",
        choices=["auto", "vis", "pyvis"],
        default="vis",
        help="HTML backend: 'vis' (robust), 'pyvis' (if it works), or 'auto' (try pyvis then fallback).",
    )
    ap.add_argument(
        "--label",
        choices=["short", "full", "none"],
        default="full",
        help="Node labels: short=leaf name, full=module.path.name, none=no labels.",
    )
    ap.add_argument(
        "--layout",
        choices=["spring", "kk", "fr"],
        default="spring",
        help="Layout algorithm for PNG: spring|kk (Kamada-Kawai)|fr (Fruchterman-Reingold).",
    )
    ap.add_argument("--dpi", type=int, default=180, help="PNG DPI (default 180).")
    ap.add_argument(
        "--json",
        action="store_true",
        help="Write graph to JSON at this path.",
    )
    ap.add_argument(
        "--json-format",
        choices=["full", "compact", "ndjson"],
        default="full",
        help="full: verbose objs; compact: index+pairs; ndjson: one object per line.",
    )
    ap.add_argument(
        "--json-no-code", action="store_true", help="Omit source code from JSON nodes."
    )
    ap.add_argument(
        "--json-code-limit",
        type=int,
        default=None,
        help="Max characters of code to include per node (truncate; default unlimited).",
    )

    return ap.parse_args()


def main():
    init_libclang()
    args = parse_args()

    include = {s.strip() for s in args.include.split(",") if s.strip()}
    for k in include:
        if k not in {"function", "struct", "global", "typedef"}:
            raise SystemExit(
                f"Unknown kind '{k}'. Allowed: function,struct,global,typedef"
            )

    extra = ["-x", "c", f"-std={args.std}", f"-I{os.path.abspath(args.project_root)}"]
    # User-specified includes and other flags
    for s in args.sys_include:
        extra += ["-isystem", s]
    extra += args.extra_args
    extra += guess_macos_sys_includes()

    builder = CodeGraphBuilder(
        project_root=args.project_root,
        include_kinds=include,
        extra_args=extra,
        prefer_headers=not args.no_headers,
        trace_args=args.trace_args,
    )
    builder.parse_all()
    builder.add_edges()
    builder.export_graphml(args.output)

    # Small, direct summary
    print(f"Nodes: {len(builder.G.nodes())}  Edges: {len(builder.G.edges())}")
    print(f"Wrote {args.output}")

    # Other outputs
    if args.json:
        if args.json_format == "full":
            export_json_full(
                builder.G,
                args.output.split(".")[0] + ".json",
                omit_code=args.json_no_code,
                code_limit=args.json_code_limit,
            )
        elif args.json_format == "compact":
            export_json_compact(
                builder.G,
                args.output.split(".")[0] + ".json",
                omit_code=args.json_no_code,
                code_limit=args.json_code_limit,
            )
        else:  # ndjson
            export_json_ndjson(
                builder.G,
                args.output.split(".")[0] + ".json",
                omit_code=args.json_no_code,
                code_limit=args.json_code_limit,
            )
        print(f"Wrote {args.json}")
    if args.png:
        save_png(
            builder.G,
            args.output.split(".")[0] + ".png",
            label_mode=args.label,
            layout=args.layout,
            dpi=args.dpi,
        )
        print(f"Wrote {args.png}")
    if args.html:
        if args.html_engine == "vis":
            export_html_visjs(
                builder.G,
                args.output.split(".")[0] + ".html",
                label_mode=args.label,
                physics=True,
            )
            print(f"Wrote {args.html}")
        elif args.html_engine == "pyvis":
            export_html_pyvis(
                builder.G,
                args.output.split(".")[0] + ".html",
                label_mode=args.label,
                physics=True,
            )
            print(f"Wrote {args.html}")
        else:  # auto
            try:
                export_html_pyvis(
                    builder.G,
                    args.output.split(".")[0] + ".html",
                    label_mode=args.label,
                    physics=True,
                )
                print(f"Wrote {args.html}")
            except Exception as e:
                print(f"[warn] PyVis failed ({e}). Falling back to vis.js…")
                export_html_visjs(
                    builder.G,
                    args.output.split(".")[0] + ".html",
                    label_mode=args.label,
                    physics=True,
                )
                print(f"Wrote {args.html}")


if __name__ == "__main__":
    main()
