# MOSAIBEADS quality verification

## V3.0.0 verification — 2026-08-12

- Automated tests: **13 passed**.
- Portable executable: `MOSAIBEADS_v3.0.exe`, **67,903,098 bytes**.
- Packaged smoke test: **exit code 0**.
- EXE SHA-256:
  `FB7B70BDDFD3D669A28613AAD9DED07F35A83980B069683EF0E4B71B8AAD8F44`.
- AI model: 3,717,007 bytes; SHA-256:
  `F405388CB07A6EAAA439DCB5D384D48A5C758C9A48E4A5B6A945CFA4363479A5`.
- Packaged GUI was opened from the final build and visually confirmed to show
  `MOSAIBEADS 3.0`, sections 3/4 in the top toolbar, sections 1/2 in the left
  sidebar, two center canvases and the right palette panel.
- A generated 16-color pattern was used for visual QA of the right panel: four
  columns of real color swatches, selected-color chip, code labels, scrollbar,
  count table and summary information were simultaneously visible.

## V3 UI invariants

The automated UI tests verify that:

- only `1 · 尺寸与色板` and `2 · 传神程度` remain in the left panel;
- `3 · 智能方案` and `4 · 预览与精修` are top-level toolbar cards;
- the former “用更少豆粒…” subtitle is absent;
- at 1360×820 the right panel keeps its full 330 px width and the center remains
  at least 650 px wide;
- the number of swatches equals the result palette size;
- selecting a swatch, painting a cell, updating counts, undoing and redoing all
  preserve the selected color and synchronized table row.

## Algorithm and export checks

The complete test suite also verifies:

- sRGB/Lab reference values and the published CIEDE2000 Sharma pair;
- all four packaged brand palettes;
- output dimensions, color limit and material count;
- thin salient-line rescue;
- strict fixed-color allow lists;
- click-color recommendation ranking;
- three switchable auto-tune intents;
- joint search coverage for palette, background, size, color budget, profile and
  likeness controls;
- locked auto-tune parameters never change;
- all six export artifacts, including PNG, PDF, CSV and JSON.

## Version separation

- `v2.3.0` freezes the last BeadSketch Studio tree and executable.
- `v3.0.0` contains the MOSAIBEADS rename, V3 interface and palette editor.
- GitHub tags/Releases and local release directories keep the two deliverables
  separate; build intermediates and old flat-directory executables are not mixed
  into a release bundle.

## Practical acceptance criteria

- A result never exceeds the selected color limit.
- The material count equals width × height after editing.
- Current paint color survives all count refreshes and undo/redo actions.
- Exported charts contain cell codes, 5-cell guides, heavier 29×29 board boundaries
  and a per-color material legend.
- The optional AI model remains local and automatically falls back to CPU or the
  non-AI scorer when unavailable.
