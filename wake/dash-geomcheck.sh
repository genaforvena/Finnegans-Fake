#!/usr/bin/env bash
# Gate for dash.sh's ONE hard property: the frame must fit its pane, at every
# width, on every render.
#
# Why this exists. `tmux capture-pane -p` is VISIBLE-ONLY. A frame one row too
# tall scrolls its own head off, so the header — which carries the clock, the
# only per-frame-changing token — never enters the hash the liveness watchdogs
# read. A pane re-rendering every 30s for 7h then reads FROZEN, gets flagged,
# dispatched, and nearly "healed" by a respawn that would have replaced the
# live dash with a bare login shell. That happened twice on 2026-08-16.
#
# Run it after ANY edit to dash.sh:  bash wake/dash-geomcheck.sh
# ~70s per geometry (it must outwait a full 30s render cadence, twice).
#
# TWO FIXTURE TRAPS, both hit while writing this, both of which made the check
# report a defect that was not there — read them before changing the harness:
#
#   1. `tmux new-session -d -x W -y H` does NOT bind the size. The server's
#      window-size option snaps the new window to the attached client's size
#      (226x56 here), so a fixture asking for 80x24 silently measured 226x56
#      and every assertion was about the wrong geometry. Pin it with
#      `window-size manual` + an explicit resize-window, and then ASSERT the
#      pane really is that size before measuring anything.
#
#   2. The resize lands AFTER the dash has already drawn frame 1 at the old
#      geometry, and the dash re-reads `tput` only once per frame. Capturing at
#      t+8s measured that stale frame and reported header_row1=no — a defect of
#      the fixture, not of the dash. Wait out a full render cadence first.
#
# Both mutation-verified 2026-08-16: reverting the header clip goes RED at
# 60x14 and stays GREEN at the live 226x27 (which is exactly why the defect
# survived the first fix); reverting the footer clip goes RED at 30x10.

set -u
D="${DASH:-$(cd "$(dirname "$0")" && pwd)/dash.sh}"
CAD="${CAD:-35}"                      # > dash.sh's 30s render cadence
GEOMS="${GEOMS:-60x14 80x24 226x27}"  # narrow, standard, and the live pane
fail=0

for geom in $GEOMS; do
  c=${geom%x*}; r=${geom#*x}; s="dashgeom-$c-$r-$$"
  tmux kill-session -t "$s" 2>/dev/null
  tmux new-session -d -s "$s" -x "$c" -y "$r" "bash $D" || { echo "$geom: cannot start"; fail=1; continue; }
  tmux set-option -t "$s" window-size manual >/dev/null
  tmux set-option -t "$s" status off >/dev/null
  tmux resize-window -t "$s" -x "$c" -y "$r" >/dev/null
  sleep "$CAD"

  real=$(tmux display -p -t "$s" '#{pane_width}x#{pane_height}')
  if [ "$real" != "$geom" ]; then     # trap 1 — never measure the wrong pane
    printf '%-8s SKIP — pane came up %s, fixture geometry not achievable\n' "$geom" "$real"
    fail=1; tmux kill-session -t "$s" 2>/dev/null; continue
  fi

  a=$(tmux capture-pane -p -t "$s"); hist_a=$(tmux display -p -t "$s" '#{history_size}')
  sleep "$CAD"
  b=$(tmux capture-pane -p -t "$s"); hist_b=$(tmux display -p -t "$s" '#{history_size}')

  head1=$(printf '%s\n' "$b" | head -1)
  lastnb=$(printf '%s\n' "$b" | grep -v '^[[:space:]]*$' | tail -1)
  ha=$(printf '%s' "$a" | cksum | cut -d' ' -f1); hb=$(printf '%s' "$b" | cksum | cut -d' ' -f1)

  hdr=no;    case "$head1"  in *'FINNEGANS FAKE'*) hdr=yes;; esac
  ftr=no;    case "$lastnb" in *'frame '*) ftr=yes;; esac
  # the footer prints the geometry the dash BELIEVES it has; a mismatch means it
  # is sizing the frame against a stale or fallback reading
  geo_ok=no; case "$lastnb" in *"${c}x${r}"*) geo_ok=yes;; esac
  # Corroboration only, NOT the discriminator. The tempting theory — "an
  # overflowing frame pushes a row into scrollback every render, so
  # history_size grows" — was tested and is FALSE: `clear` resets the screen
  # without spilling into history, and the pre-fix dash measured hist_static
  # at the live 226x27 while overflowing every frame. (The hist=1/hist=7 seen
  # while diagnosing came from the resize, not the overflow.) header_row1 is
  # the property: if the header is visible, nothing scrolled off.
  [ "$hist_a" = "$hist_b" ] && grow=no || grow=yes
  # `advances` is the SYMPTOM the watchdog sees, not the property under test:
  # the pre-fix dash scored advances=yes on one run and no on another, purely
  # on whether an nvidia-smi digit in the surviving tail happened to move.
  [ "$ha" != "$hb" ] && adv=yes || adv=no

  printf '%-8s header_row1=%-3s footer_last=%-3s footer_geom=%-3s hist_static=%-3s advances=%s\n' \
    "$geom" "$hdr" "$ftr" "$geo_ok" "$([ "$grow" = no ] && echo yes || echo no)" "$adv"
  [ "$hdr" = yes ] && [ "$ftr" = yes ] && [ "$geo_ok" = yes ] && [ "$grow" = no ] && [ "$adv" = yes ] || fail=1
  tmux kill-session -t "$s" 2>/dev/null
done

if [ "$fail" = 0 ]; then echo "dash-geomcheck: ok"; else echo "dash-geomcheck: FAIL"; fi
exit "$fail"
