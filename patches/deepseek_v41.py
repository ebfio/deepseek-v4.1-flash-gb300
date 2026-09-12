# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 reasoning and spaced DSML tool calls.
Patched build (dsv41-sm103): streaming-safe DSML recovery. A malformed V4.1
marker is repaired instead of leaking into ``content`` as raw markup.
"""

import functools
from dataclasses import replace

import regex as re

from vllm.parser.deepseek_v4 import (
    DeepSeekV4Parser,
    _dsml_arg_converter,
    deepseek_v4_config,
)
from vllm.parser.engine.parser_engine_config import ParserEngineConfig

DSML_TOOL_START = "<｜DSML｜ calls>"
DSML_TOOL_END = "</｜DSML｜ calls>"
DSML_INVOKE_PREFIX = '<｜DSML｜ invoke name="'
DSML_INVOKE_END = "</｜DSML｜ invoke>"
DSML_PARAM_START = "<｜DSML｜ parameter"
DSML_PARAM_CLOSE = "</｜DSML｜ parameter>"

_PARAM_VALUE = r"((?:(?!</?｜).)*)"
_PARAM_END = r"(?:</｜[^<>]*>?|(?=<｜))"
_PARAM_RE = re.compile(
    r'<｜DSML｜ parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?>'
    + _PARAM_VALUE + _PARAM_END,
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    r'<｜DSML｜ parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?>'
    + _PARAM_VALUE,
    re.DOTALL,
)


@functools.cache
def deepseek_v41_config(thinking: bool = False) -> ParserEngineConfig:
    config = deepseek_v4_config(thinking=thinking)
    terminals = config.terminals | {
        "TOOL_START": DSML_TOOL_START,
        "TOOL_END": DSML_TOOL_END,
        "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
        "INVOKE_END": DSML_INVOKE_END,
        "PARAM_START": DSML_PARAM_START,
        "PARAM_CLOSE": DSML_PARAM_CLOSE,
    }
    return replace(
        config,
        name="deepseek_v41",
        terminals=terminals,
        token_id_terminals={key: terminals[key] for key in config.token_id_terminals},
        arg_converter=functools.partial(
            _dsml_arg_converter,
            param_re=_PARAM_RE,
            partial_param_re=_PARTIAL_PARAM_RE,
        ),
    )


class DeepSeekV41Parser(DeepSeekV4Parser):
    parser_config = staticmethod(deepseek_v41_config)
# ---------------------------------------------------------------------------
# DSML recovery (dsv41-sm103 patch) -- streaming-safe
# ---------------------------------------------------------------------------
#
# DeepSeek's reference parser documents that the model "might occasionally
# generate" malformed output and explicitly declines to recover (model repo,
# encoding/README.md): callers are told to add their own error handling. vLLM's
# v4.1 engine is a strict state machine with no recovery: on a malformed marker
# it leaks the raw DSML into ``content`` and drops the call.
#
# Malformations seen in production:
#   1. unspaced V4 tags (<|DSML|calls>) -- V4.1 moved to a leading space, the
#      lexer anchors on the spaced spelling and never matches;
#   2. missing invoke wrapper -- the tool name is emitted in a bare
#      <parameter name="X"> line instead of <|DSML| invoke name="X">.
#
# WHY THIS IS NOT A PER-DELTA REGEX: ``_preprocess_feed`` is called once per
# streaming delta and a pattern like
#
#     <parameter name="bash">\n<|DSML| parameter ...
#
# never lands in a single delta -- the server emits a few tokens at a time. A
# plain ``re.sub`` on ``delta_text`` matches nothing in production while passing
# any test that feeds the block as one chunk. The repair therefore runs on a
# rolling buffer with a holdback, so a pattern split across deltas is seen whole
# before any of it is emitted.

_D = chr(0xFF5C)
_TAG = "<" + _D + "DSML" + _D
_TAG_CLOSE = "</" + _D + "DSML" + _D

# Markers we must see whole before emitting the text that precedes them.
# The plain-ASCII wrapper openers are included so the fully degraded dialect
# (wrapper tags degraded too) is held from its first line instead of streaming
# out as content before the block is recognised.
_MARKERS = (
    "<parameter name=",
    "<tool_calls>",
    "<invoke name=",
    _TAG,
    "</" + _D + "DSML" + _D,
)

# A bare <parameter name="X">\n line, as emitted by the missing-invoke case.
_PARAM_LINE_RE = re.compile(r'<parameter\s+name="[^"]*">[ \t]*\n[ \t]*$')

# Upper bound on held characters, so malformed-looking prose cannot stall output.
_MAX_HOLD = 512

# <|DSML|calls> -> <|DSML| calls>  (and the closing form)
_UNSPACED_RE = re.compile(
    r"(</?" + _D + r"[A-Za-z_]{0,14}" + _D + r")(?=[A-Za-z_])"
)

# <parameter name="X">\n<|DSML| parameter -> <|DSML| invoke name="X">\n<|DSML| parameter
_MISSING_INVOKE_RE = re.compile(
    r'<parameter\s+name="([^"]+)">[ \t]*\n(?=' + re.escape(_TAG) + r"\s*parameter)"
)
_MISSING_INVOKE_SUB = _TAG + ' invoke name="\\1">' + chr(10)
# ---------------------------------------------------------------------------
# Pure XML dialect normalizer: the fully degraded shape.
# ---------------------------------------------------------------------------
#
# Production runs show the model collapsing the WHOLE marker set to plain
# tags with no DSML sigil and no string="..." attribute:
#
#   <parameter name="bash">            <- invoke selector (bare line)
#   <parameter name="command">echo M   <- one string-typed param (value inline)
#   </parameter>                       <- closes param
#
# The lexer state machine only understands the full DSML spelling, so these
# lines must be rewritten. Run this BEFORE the generic recovery when a bare
# XML parameter line appears, and only for lines that are actually inside a
# contiguous block of the same dialect (a bare "</parameter>" in prose is
# harmless to rewrite since it only opens/closes nothing without a start).

_XML_PARAM_OPEN = re.compile(
    r'<parameter\s+name="([^"]+)"(\s+string="(true|false)")?>'
)
# bare open with nothing after on the line = invoke selector
_XML_OPEN_LINE_RE = re.compile(
    r'(?m)^<parameter\s+name="([^"]+)"\s*>\s*$'
)
# inline open + value = parameter line
_XML_PARAM_LINE_RE = re.compile(
    r'(?m)^<parameter\s+name="([^"]+)"(\s+string="(true|false)")?>(.*?)(?:\r?\n|$)'
)
# A parameter closer in ANY dialect: bare, single-sigil, or full DSML. The
# single-sigil form ("</" + sigil + "parameter>") is the common production
# shape and carries no DSML word, so a bare-"</parameter>"-only test misses it.
_XML_CLOSER_RE = re.compile(
    r"</(?:parameter>"
    r"|" + re.escape(_D) + r"parameter>"
    r"|" + re.escape(_D) + r"DSML" + re.escape(_D) + r"\s*parameter>)"
)
# Plain-ASCII wrapper tags: the dialect degraded all the way through the
# tool_calls/invoke levels, with no sigil anywhere. These carry the block's
# structure, so they are rewritten to DSML rather than passed through.
_XML_TOOLCALLS_OPEN_RE = re.compile(r"(?m)^<tool_calls>\s*$")
# The calls closer may be followed on the same line by the EOS token
# ("</tool_calls></assistant>"), so anchor on the tag itself rather than on
# end-of-line: an end-of-line anchor never matches there and the whole block
# stays held until the stream ends, which is exactly the leak this recovers.
_XML_TOOLCALLS_CLOSE_RE = re.compile(r"</tool_calls>")
_XML_INVOKE_OPEN_RE = re.compile(r'(?m)^<invoke\s+name="([^"]+)"\s*>\s*$')
_XML_INVOKE_CLOSE_RE = re.compile(r"</invoke>")
# The first line of any sigilless XML block: wrapper opener or a bare parameter
# opener. Used by the holdback to know where the block starts.
_XML_BLOCK_START_RE = re.compile(
    r'(?m)^(?:<tool_calls>\s*$'
    r'|<invoke\s+name="[^"]+"\s*>\s*$'
    r'|<parameter\s+name="[^"]+"(?:\s+string="(true|false)")?>\s*$)'
)

def _xml_lines_to_dsml(text: str) -> str:
    """Rewrite a contiguous XML-dialect tool block into DSML."""
    # Engage whenever a sigilless "<parameter name=" opener OR a plain-ASCII
    # wrapper opener is present, even if a DSML sigil appears elsewhere in the
    # buffer. Production emits sigilless openers whose closer DOES carry the
    # sigil, so a whole-buffer sigil test refuses exactly the shape that leaks.
    # Lines already in DSML are passed through untouched (see the loop), so
    # mixed buffers never double-convert.
    if not (_XML_PARAM_OPEN.search(text)
            or _XML_TOOLCALLS_OPEN_RE.search(text)
            or _XML_INVOKE_OPEN_RE.search(text)):
        return text

    lines = text.split("\n")
    out = []
    saw_invoke = False
    seen_closer = False
    for ln in lines:
        # Pass through lines that are ALREADY DSML, so a mixed buffer never
        # double-converts (e.g. the missing-invoke repair's own output).
        # Matching on the full marker, not on the sigil alone, is essential: a
        # sigilless opener whose INLINE closer carries the sigil contains _D but
        # is not DSML, and must still be converted below.
        if ln.startswith(_TAG) or ln.startswith(_TAG_CLOSE):
            if ln.startswith(_TAG_CLOSE):
                seen_closer = True
            out.append(ln)
            continue
        # Plain-ASCII wrapper tags (fully degraded dialect). The invoke line
        # carries the tool name, so it establishes the invoke just like a bare
        # parameter selector line would.
        if _XML_TOOLCALLS_OPEN_RE.match(ln):
            out.append(_TAG + " calls>")
            continue
        miv = _XML_INVOKE_OPEN_RE.match(ln)
        if miv and not saw_invoke:
            out.append(_TAG + ' invoke name="' + miv.group(1) + '">')
            saw_invoke = True
            continue
        if (_XML_TOOLCALLS_CLOSE_RE.match(ln)
                or _XML_INVOKE_CLOSE_RE.match(ln)
                or _XML_TOOLCALLS_CLOSE_RE.search(ln)
                or _XML_INVOKE_CLOSE_RE.search(ln)):
            # Consumed here, not re-emitted: the DSML closers are synthesized at
            # the end of this function. Appending the raw ASCII closers would
            # leak them into content after the block converted. ``search`` (not
            # ``match``) because the closer can share a line with the EOS token.
            continue
        m = _XML_OPEN_LINE_RE.match(ln)
        if m and not saw_invoke:
            out.append(_TAG + ' invoke name="' + m.group(1) + '">')
            saw_invoke = True
            continue
        mp = _XML_PARAM_LINE_RE.match(ln)
        if mp and saw_invoke:
            nm = mp.group(1)
            st = mp.group(2) or ' string="true"'
            raw = mp.group(4)
            # The closer is often written INLINE, at the end of the value line:
            #   <parameter name="command">echo hi</parameter>
            # It must be removed from the value here. It carries no DSML sigil,
            # so the engine's value regex would otherwise swallow it into the
            # argument (the production leak: args ended with "</parameter>").
            cm = _XML_CLOSER_RE.search(raw)
            if cm:
                val = raw[:cm.start()].rstrip()
                tail = raw[cm.end():]
                out.append(_TAG + ' parameter name="' + nm + '"' + st + '>' + val)
                out.append("</" + _D + "DSML" + _D + "parameter>")
                if tail.strip():
                    out.append(tail)
                seen_closer = True
                continue
            val = raw.rstrip()
            out.append(_TAG + ' parameter name="' + nm + '"' + st + '>' + val)
            continue
        if _XML_CLOSER_RE.match(ln) and saw_invoke:
            out.append("</" + _D + "DSML" + _D + "parameter>")
            seen_closer = True
            continue
        if ln.startswith("</") or ln.startswith("<") :
            # A partial/foreign tag: stop converting; keep the rest as-is so a
            # later delta can finish the block.
            out.append(ln)
            if ln.startswith("</") and not ln.startswith("</parameter>"):
                pass
            continue
        out.append(ln)
    if not (saw_invoke and seen_closer):
        # The block is not complete yet (the closer "</parameter>" has not
        # arrived). Return the ORIGINAL text so the caller's holdback keeps
        # holding; converting here would emit an incomplete tool call.
        return text
    out_text = "\n".join(out)
    out_text += "\n" + "</" + _D + "DSML" + _D + " invoke>"
    out_text += "\n" + "</" + _D + "DSML" + _D + " calls>"
    return out_text
def _dsml_recover(text):
    """Repair malformed V4.1 DSML markup. Idempotent; ASCII fast path."""
    if _D in text:
        text = _UNSPACED_RE.sub(lambda m: m.group(1) + " ", text)
        text = _MISSING_INVOKE_RE.sub(_MISSING_INVOKE_SUB, text)
    # Run the XML normalizer last, unconditionally. The unspacing above rewrites
    # a sigil'd closer into the spaced spelling the state machine recognises,
    # and the normalizer is a no-op unless a sigilless opener is present, so a
    # pure-DSML buffer still passes through unchanged.
    return _xml_lines_to_dsml(text)


def _holdback_len(buf):
    """Number of trailing chars that must not be emitted yet."""
    # XML-dialect block: a sigilless opener or plain-ASCII wrapper opener is
    # present. The normalizer needs the WHOLE block (open selector -> params ->
    # closer) in one buffer, so hold from the first such line until the block is
    # closed. The sigil test is deliberately NOT applied here: production closers
    # carry the sigil even when the openers do not, so requiring a sigil-free
    # buffer would let exactly this shape stream past unrecovered.
    mb = _XML_BLOCK_START_RE.search(buf)
    if mb:
        first = mb.start()
        # A wrapper-led block is only complete once its calls closer arrives.
        # Stopping at the first parameter closer would convert and emit the head
        # of the block, leaving the trailing "</invoke></tool_calls>" to stream
        # out as content afterwards.
        if _XML_TOOLCALLS_OPEN_RE.search(buf):
            complete = bool(_XML_TOOLCALLS_CLOSE_RE.search(buf))
        else:
            complete = bool(_XML_CLOSER_RE.search(buf))
        if complete:
            return 0  # block complete: recovery will convert it
        # Incomplete block: hold everything from the first open, bounded so a
        # prose mention of a tag cannot stall output indefinitely.
        if len(buf) - first <= _MAX_HOLD * 2:
            return len(buf) - first
        return 0

    last = buf.rfind("<")
    if last == -1:
        return 0
    tail = buf[last:]
    if len(tail) > _MAX_HOLD:
        return 0

    partial = False
    for marker in _MARKERS:
        if marker.startswith(tail):      # tail could still become a marker
            partial = True
            break
        if tail.startswith(marker):      # complete marker; rest is unresolved
            partial = True
            break
    if not partial:
        return 0

    start = last
    # The missing-invoke repair spans two lines, so if a bare
    # <parameter name="X"> line sits immediately before the marker, hold it too.
    m = _PARAM_LINE_RE.search(buf, 0, last)
    if m:
        start = m.start()
    return len(buf) - start


def _decode(tokenizer, ids):
    if not ids or tokenizer is None:
        return ""
    try:
        out = tokenizer.decode(list(ids))
    except Exception:
        return ""
    return out if isinstance(out, str) else ""


def _split_ids(tokenizer, ids, want_chars):
    """Split ids so the first part decodes to about want_chars characters."""
    if not ids or want_chars <= 0 or tokenizer is None:
        return [], list(ids)
    if len(_decode(tokenizer, ids)) <= want_chars:
        return list(ids), []
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(_decode(tokenizer, ids[:mid])) <= want_chars:
            lo = mid
        else:
            hi = mid - 1
    return list(ids[:lo]), list(ids[lo:])


def _reencode(tokenizer, text):
    """Encode text to ids for the engine, tolerating tokenizer absence."""
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return []


def _preprocess_feed(self, delta_text, delta_token_ids):
    """Engine hook: see ParserEngine._preprocess_feed.

    Called once per streaming delta, so the repair runs over a rolling buffer
    with a holdback rather than over the delta alone.
    """
    if not delta_text:
        return delta_text, delta_token_ids

    buf = getattr(self, "_dsml_buf", "") + delta_text
    ids = list(getattr(self, "_dsml_ids", [])) + list(delta_token_ids)
    hold = _holdback_len(buf)

    if hold >= len(buf):
        # Everything is still an unresolved marker prefix; buffer and emit none.
        self._dsml_buf, self._dsml_ids = buf, ids
        return "", []

    if hold == 0:
        self._dsml_buf, self._dsml_ids = "", []
        return _dsml_recover(buf), ids

    emit_text = buf[:-hold]
    emit_ids, held_ids = _split_ids(self.model_tokenizer, ids, len(emit_text))
    self._dsml_buf = buf[len(emit_text):]
    self._dsml_ids = held_ids
    return _dsml_recover(emit_text), emit_ids

_V41_RESET = DeepSeekV41Parser._reset


def _reset(self, initial_state=None):
    _V41_RESET(self, initial_state)
    self._dsml_buf = ""
    self._dsml_ids = []


_V41_FINISH = DeepSeekV41Parser.finish_streaming


def _finish_streaming(self):
    """Flush any text still held by the recovery buffer before finishing.

    Without this, a stream that ends while a marker prefix is held back (e.g.
    the last emitted char is "<") would silently drop that tail.
    """
    held = getattr(self, "_dsml_buf", "")
    if held:
        self._dsml_buf, self._dsml_ids = "", []
        try:
            self._engine.feed(_dsml_recover(held), [])
        except Exception:
            pass
    return _V41_FINISH(self)


DeepSeekV41Parser._preprocess_feed = _preprocess_feed
DeepSeekV41Parser._reset = _reset
DeepSeekV41Parser.finish_streaming = _finish_streaming
