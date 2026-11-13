#!/bin/bash
# --- laion v2: 4-tier forgiving line-completion verifier ---
# tiers: 1 exact / 2 agent-startswith-gold / 3 leading-identifier / 4 non-empty-partial(0.25)
set -u
mkdir -p /logs/verifier
echo "0" > /logs/verifier/reward.txt          # DEFAULT-FAIL
echo "0.0" > /logs/verifier/pass_ratio.txt

AGENT_OUT=/app/solution.txt
GOLD_OUT=/tests/solution.txt
if [ ! -f "$AGENT_OUT" ]; then
    { echo "FAIL: agent did not write $AGENT_OUT"; } | tee /logs/verifier/test_output.txt
    python3 -m pytest /tests/test_state.py -q -p no:cacheprovider >/dev/null 2>&1 || true
    exit 1
fi
if [ ! -f "$GOLD_OUT" ]; then
    { echo "FAIL: gold $GOLD_OUT missing"; } | tee /logs/verifier/test_output.txt
    python3 -m pytest /tests/test_state.py -q -p no:cacheprovider >/dev/null 2>&1 || true
    exit 1
fi

normalise() {
    sed -e 's/\r$//' -e 's/[ \t]*$//' "$1" \
        | awk 'BEGIN{p=0} /[^ \t]/{p=1} p{print}' | tac \
        | awk 'BEGIN{p=0} /[^ \t]/{p=1} p{print}' | tac
}
A=$(mktemp); B=$(mktemp)
normalise "$AGENT_OUT" > "$A"; normalise "$GOLD_OUT" > "$B"

reward="0"; ratio="0.0"; tier=""
if diff -q "$A" "$B" >/dev/null 2>&1; then
    reward="1"; ratio="1.0"; tier="1/exact"
else
    GL=$(wc -l < "$B")
    if [ "$GL" -gt 0 ]; then
        AH=$(mktemp); head -n "$GL" "$A" > "$AH"
        if diff -q "$AH" "$B" >/dev/null 2>&1; then
            EXTRA=$(($(wc -l < "$A") - GL))
            if [ "$EXTRA" -le 3 ]; then reward="1"; ratio="1.0"; tier="2/startswith"; fi
        fi
        rm -f "$AH"
    fi
fi
if [ "$reward" = "0" ]; then
    fid() { grep -oE '[A-Za-z_][A-Za-z0-9_]*' "$1" 2>/dev/null | head -n 1; }
    AI=$(fid "$A"); BI=$(fid "$B")
    if [ -n "$AI" ] && [ -n "$BI" ] && [ "$AI" = "$BI" ]; then
        reward="1"; ratio="1.0"; tier="3/leading-identifier"
    fi
fi
if [ "$reward" = "0" ]; then
    SZ=$(wc -c < "$A"); IC=$(grep -oE '[A-Za-z_][A-Za-z0-9_]*' "$A" 2>/dev/null | wc -l)
    if [ "$SZ" -gt 0 ] && [ "$SZ" -le 400 ] && [ "$IC" -ge 1 ]; then
        reward="0.25"; ratio="0.25"; tier="4/non-empty-partial"
    fi
fi

echo "$reward" > /logs/verifier/reward.txt
echo "$ratio" > /logs/verifier/pass_ratio.txt
{ echo "PASS[$tier] (reward=$reward ratio=$ratio)"; echo "--- gold ---"; cat "$B";
  echo "--- agent ---"; cat "$A"; } | tee /logs/verifier/test_output.txt
rm -f "$A" "$B"
python3 -m pytest /tests/test_state.py -q -p no:cacheprovider >/dev/null 2>&1 || true
exit 0
