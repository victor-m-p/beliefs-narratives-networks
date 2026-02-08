(function (w) {
  'use strict';

  // ---------- shared helpers ----------
  function wrapText(ctx, text, maxWidth) {
    const words = String(text || '').split(/\s+/);
    const lines = [];
    let line = '';
    for (const w of words) {
      const test = line ? line + ' ' + w : w;
      if (ctx.measureText(test).width <= maxWidth) line = test;
      else {
        if (line) lines.push(line);
        line = w;
      }
    }
    if (line) lines.push(line);
    return lines;
  }

  const overlap = (a, b) =>
    !(
      a.right <= b.left ||
      b.right <= a.left ||
      a.bottom <= b.top ||
      b.bottom <= a.top
    );

  // ---------- defaults (same look as your original) ----------
  const DRAW_DEFAULTS = {
    borderColor: '#000',
    borderWidth: 3,
    labelFont: '14px sans-serif',
    labelColor: '#111',
    labelLineHeight: 18,
    labelMaxWidth: 180,
    labelGap: 4, // gap from circle
    node: {
      defaultRadius: 15,
      orange: '#5CA4A9',
      orangeSelected: '#8CC7CA',
      grey: '#999999',
      greySelected: '#777777',
      flash: '#ff4d4d',
      stroke: 'transparent', // '#000000',
      strokeWidth: 0, // #2,
    },
    edge: {
      positive: 'green',
      negative: 'red',
      other: '#666666',
      minWidth: 1,
      maxWidth: 10,
    },
    dodge: {
      passes: 22, // solver iterations
      maxShift: 70, // max horizontal dodge (px)
      separatePad: 2, // extra px when separating
      preferAbovePenalty: 0.6, // bias: flipping below must reduce overlaps by at least this margin
    },
  };
  // tune these once, use everywhere
  const UI_DEFAULTS = {
    dragThresholdPx: 6, // movement needed before we start dragging
    longPressMs: 180, // or drag after this time holding down (optional)
  };

  // inserting new thing for hover-highlight
  const HOVERED = new WeakMap();

  // ---------- measure label block for a node (position = 'above' | 'below') ----------
  function measureLabel(ctx, p, opts, position = 'above') {
    const o = { ...DRAW_DEFAULTS, ...opts };
    const n = { ...DRAW_DEFAULTS.node, ...(opts?.node || {}) };
    const r = p.radius ?? n.defaultRadius;

    ctx.font = o.labelFont;
    const lines = wrapText(ctx, String(p.label || ''), o.labelMaxWidth);
    let widest = 0;
    for (const ln of lines)
      widest = Math.max(widest, ctx.measureText(ln).width);

    const w = widest;
    const h = lines.length * o.labelLineHeight;

    let top;
    if (position === 'above') {
      const blockBottomY = p.y - r / 1.5 - o.labelGap;
      top = blockBottomY - h;
    } else {
      // position 'below': sit just under the circle
      top = p.y + r + o.labelGap;
    }

    const leftCentered = (xCenter) => Math.round(xCenter - w / 2);
    return {
      lines,
      w,
      h,
      top,
      leftCentered,
      lineHeight: o.labelLineHeight,
      position,
    };
  }

  // ---------- compute overlaps for a candidate rect ----------
  function rectFor(rec, dx, position) {
    const m = rec.mByPos[position];
    const left = m.leftCentered(rec.p.x + dx);
    return { left, right: left + m.w, top: m.top, bottom: m.top + m.h };
  }

  function totalOverlapsFor(index, info, dxs, posStates) {
    let count = 0;
    const a = rectFor(info[index], dxs[index], posStates[index]);
    for (let j = 0; j < info.length; j++) {
      if (j === index) continue;
      const b = rectFor(info[j], dxs[j], posStates[j]);
      if (overlap(a, b)) count++;
    }
    return count;
  }

  // ---------- dodge solver (horizontal + optional flip below) ----------
  function computeDodges(ctx, canvas, points, opts) {
    const o = { ...DRAW_DEFAULTS, ...opts };
    const dset = o.dodge;

    // Build records with measurements for both positions
    const info = points.map((p) => {
      const mAbove = measureLabel(ctx, p, o, 'above');
      const mBelow = measureLabel(ctx, p, o, 'below');
      return { p, mByPos: { above: mAbove, below: mBelow } };
    });

    // State: per label horizontal shift and position
    const dxs = info.map(() => 0);
    const posStates = info.map(() => 'above'); // default above

    // Iterative resolve
    for (let it = 0; it < dset.passes; it++) {
      let moved = false;

      // 1) Try flipping some labels below if it reduces conflicts noticeably
      for (let i = 0; i < info.length; i++) {
        // 👇 respect "lock above" (e.g., central 'Meat Eating')
        if (posStates[i] === 'below' || info[i].p._lockAbove) continue;

        const currentOver = totalOverlapsFor(i, info, dxs, posStates);
        if (currentOver === 0) continue;

        const before = currentOver;
        const oldPos = posStates[i];
        posStates[i] = 'below';
        const after = totalOverlapsFor(i, info, dxs, posStates);

        // keep the flip only if it clearly helps
        if (!(after + dset.preferAbovePenalty < before)) {
          posStates[i] = oldPos; // revert
        } else {
          moved = true;
        }
      }

      // 2) Horizontal push apart overlapping pairs
      for (let i = 0; i < info.length; i++) {
        for (let j = i + 1; j < info.length; j++) {
          const aRect = rectFor(info[i], dxs[i], posStates[i]);
          const bRect = rectFor(info[j], dxs[j], posStates[j]);
          if (!overlap(aRect, bRect)) continue;

          const xOverlap =
            Math.min(aRect.right, bRect.right) -
            Math.max(aRect.left, bRect.left);
          const shift = Math.ceil(xOverlap / 2) + dset.separatePad;

          if (info[i].p.x <= info[j].p.x) {
            dxs[i] -= shift;
            dxs[j] += shift;
          } else {
            dxs[i] += shift;
            dxs[j] -= shift;
          }

          dxs[i] = Math.max(-dset.maxShift, Math.min(dset.maxShift, dxs[i]));
          dxs[j] = Math.max(-dset.maxShift, Math.min(dset.maxShift, dxs[j]));
          moved = true;
        }
      }

      if (!moved) break;
    }

    // return maps for drawing
    return {
      dxMap: new Map(info.map((rec, i) => [rec.p, dxs[i]])),
      posMap: new Map(info.map((rec, i) => [rec.p, posStates[i]])),
      mByPosMap: new Map(info.map((rec) => [rec.p, rec.mByPos])),
    };
  }

  // ---------- base drawing (same as your original) ----------
  function clearAndFrame(ctx, canvas, opts = {}) {
    const o = { ...DRAW_DEFAULTS, ...opts };
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = o.borderColor;
    ctx.lineWidth = o.borderWidth;
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  }

  function drawEdges(ctx, edges = [], opts = {}) {
    const e = { ...DRAW_DEFAULTS.edge, ...(opts.edge || {}) };
    for (const edge of edges || []) {
      const w =
        e.minWidth + ((edge.strength ?? 50) / 100) * (e.maxWidth - e.minWidth);
      ctx.beginPath();
      ctx.moveTo(edge.from.x, edge.from.y);
      ctx.lineTo(edge.to.x, edge.to.y);

      if (edge.polarity === 'negative') {
        ctx.strokeStyle = e.negative;
      } else if (edge.polarity === 'other') {
        ctx.strokeStyle = e.other || '#666666'; // default grey if not passed
      } else {
        ctx.strokeStyle = e.positive;
      }

      ctx.lineWidth = w;
      ctx.stroke();
    }
  }

  function drawNode(ctx, p, selectedPoint = null, opts = {}) {
    const n = { ...DRAW_DEFAULTS.node, ...(opts.node || {}) };
    const r = p.radius ?? n.defaultRadius;

    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);

    let fill;
    if (p.flash) fill = n.flash;
    else if (p === selectedPoint || p.selected)
      fill =
        p.fixed || p.label === 'Meat Eating'
          ? n.greySelected
          : n.orangeSelected;
    else if (p.fixed || p.label === 'Meat Eating') fill = n.grey;
    else fill = n.orange;

    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = n.stroke;
    ctx.lineWidth = n.strokeWidth;
    ctx.stroke();
  }

  // draw a label at measured position (above/below) with optional dx
  function drawLabelAt(ctx, p, m, dx, opts = {}, highlight = false) {
    const o = { ...DRAW_DEFAULTS, ...opts };
    ctx.font = highlight ? `bold ${o.labelFont}` : o.labelFont; // new change
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const xCenter = p.x + (dx || 0);
    ctx.fillStyle = o.labelColor;
    for (let i = 0; i < m.lines.length; i++) {
      const y = m.top + i * m.lineHeight;
      ctx.fillText(m.lines[i], xCenter, y);
    }
  }

  // full draw: nodes, then labels with dodge+flip
  function drawGraph(ctx, canvas, data, opts = {}) {
    clearAndFrame(ctx, canvas, opts);
    if (data.edges && data.edges.length) drawEdges(ctx, data.edges, opts);
    (data.points || []).forEach((p) =>
      drawNode(ctx, p, data.selectedPoint, opts)
    );

    const points = data.points || [];
    // keep central/fixed labels always above
    points.forEach((p) => {
      if (p.fixed || p.label === 'Meat Eating') p._lockAbove = true;
    });
    if (!points.length) return;

    const { dxMap, posMap, mByPosMap } = computeDodges(
      ctx,
      canvas,
      points,
      opts
    );

    const hovered = HOVERED.get(canvas);
    points.forEach((p) => {
      const dx = dxMap.get(p) || 0;
      const pos = posMap.get(p) || 'above';
      const m = mByPosMap.get(p)[pos];

      // bold if hovered OR selected (either via data.selectedPoint or p.selected flag)
      const isSelected =
        (data && data.selectedPoint && data.selectedPoint === p) ||
        !!p.selected;

      drawLabelAt(ctx, p, m, dx, opts, hovered === p || isSelected);
    });
  }

  // ---------- interaction helpers (unchanged) ----------
  function isTooClose(x, y, points, except = null, minDistance = 35) {
    for (const p of points) {
      if (p === except) continue;
      const dx = p.x - x,
        dy = p.y - y;
      if (Math.hypot(dx, dy) < minDistance) return true;
    }
    return false;
  }
  function tryMove(point, x, y, points, draw, minDistance = 35, onReject) {
    if (!isTooClose(x, y, points, point, minDistance)) {
      point.x = x;
      point.y = y;
      if (typeof draw === 'function') draw();
      return true;
    }
    if (typeof onReject === 'function') onReject(point);
    return false;
  }
  function flash(point, draw, duration = 150) {
    point.flash = true;
    if (typeof draw === 'function') draw();
    setTimeout(() => {
      point.flash = false;
      if (typeof draw === 'function') draw();
    }, duration);
  }
  function findPoint(
    pos,
    points,
    fallbackRadius = DRAW_DEFAULTS.node.defaultRadius
  ) {
    return points.find((p) => {
      const r = p.radius ?? fallbackRadius;
      return Math.hypot(p.x - pos.x, p.y - pos.y) <= r;
    });
  }

  // inserting new here
  // set or clear the hovered point for a canvas; returns true if changed
  function setHovered(canvas, point) {
    const prev = HOVERED.get(canvas) || null;
    if (prev === point) return false;
    if (point) HOVERED.set(canvas, point);
    else HOVERED.delete(canvas);
    return true;
  }

  function clearHovered(canvas) {
    return setHovered(canvas, null);
  }

  // ---------- export (same surface as canvasfunc.js) ----------
  w.NoOverlap = {
    DRAW_DEFAULTS,
    clearAndFrame,
    drawEdges,
    drawNode,
    drawLabel: (ctx, p, opts) => {
      // legacy: draw "above"
      const m = measureLabel(ctx, p, opts, 'above');
      drawLabelAt(ctx, p, m, 0, opts);
    },
    drawGraph,
    isTooClose,
    tryMove,
    flash,
    findPoint,
    setHovered,
    clearHovered,
    UI: UI_DEFAULTS,
  };
})(window);
