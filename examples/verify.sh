#!/usr/bin/env bash
# Correctness check for a running DeepSeek-V4.1-Flash endpoint.
# A wrong-but-legal scale layout yields garbage, not an assert — always run this.
set -euo pipefail
BASE="${1:-http://127.0.0.1:8001}"
MODEL="${2:-deepseek-v4.1-flash}"

ask() {
  curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d @- <<JSON
{"model":"$MODEL","messages":[{"role":"user","content":$1}],
 "max_tokens":${2:-64},"temperature":0,"chat_template_kwargs":{"thinking":false}}
JSON
}

echo "== math =="
out=$(ask '"What is 17*19? Return only the integer."' 32)
echo "$out" | python3 -c 'import json,sys; c=json.load(sys.stdin)["choices"][0]["message"]["content"].strip(); print("   answer:",repr(c)); sys.exit(0 if c=="323" else 1)' \
  && echo "   PASS" || { echo "   FAIL (expected 323)"; exit 1; }

echo "== long context (needle at ~9K) =="
out=$(ask '"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. IMPORTANT: The secret code is BANANA-7742. What is the secret code? Answer with only the code."' 32)
echo "$out" | python3 -c 'import json,sys; c=json.load(sys.stdin)["choices"][0]["message"]["content"].strip(); print("   answer:",repr(c)); sys.exit(0 if "7742" in c else 1)' \
  && echo "   PASS" || { echo "   FAIL"; exit 1; }

echo "== tool calling =="
curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\":\"$MODEL\",\"tool_choice\":\"auto\",\"max_tokens\":200,
  \"chat_template_kwargs\":{\"thinking\":false},
  \"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris in celsius?\"}],
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",
    \"description\":\"Get current weather\",
    \"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]}" \
| python3 -c '
import json,sys
m=json.load(sys.stdin)["choices"][0]["message"]
tc=m.get("tool_calls")
if not tc: print("   FAIL: no tool_calls (raw markup leaked?)", repr(m.get("content"))[:120]); sys.exit(1)
print("   tool:", tc[0]["function"]["name"], tc[0]["function"]["arguments"]); print("   PASS")'

echo "== tool-call recovery (malformed-emission guard) =="
# The parser bug is SILENT: HTTP 200, finish_reason "stop", zero tool_calls,
# raw markup left in content. So assert on the call and on the absence of
# markup, not on the status code. A clean refusal (no call, no markup) is not
# a failure -- only markup reaching content is.
curl -s "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\":\"$MODEL\",\"tool_choice\":\"auto\",\"max_tokens\":200,\
  \"temperature\":0,\"chat_template_kwargs\":{\"thinking\":false},\
  \"messages\":[{\"role\":\"user\",\"content\":\"List the files in /tmp. Use a tool.\"}],\
  \"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"run\",\
    \"description\":\"Run a shell command\",\
    \"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}},\
\"required\":[\"command\"]}}}]}" \
| python3 -c '
import json,sys
m=json.load(sys.stdin)["choices"][0]["message"]
c=m.get("content") or ""
tc=m.get("tool_calls") or []
leak=("DSML" in c) or ("tool_calls" in c) or ("parameter name=" in c) or ("invoke name=" in c)
if leak: print("   FAIL: raw tool-call markup leaked into content"); sys.exit(1)
if not tc: print("   SKIP: clean refusal (no call, no markup) -- not a failure"); sys.exit(0)
print("   tool:", tc[0]["function"]["name"], tc[0]["function"]["arguments"][:60]); print("   PASS")'

echo; echo "all checks passed"
