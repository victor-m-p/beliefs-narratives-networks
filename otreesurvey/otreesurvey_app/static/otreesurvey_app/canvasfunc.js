// static/otreesurvey_app/canvasfunc.js
(function (w) {
  'use strict';

  // --- helpers ---
  function wrapText(ctx, text, maxWidth) {
    const words = String(text || '').split(/\s+/);
    const lines = [];
    let line = '';
    for (const w of words) {
      const test = line ? line + ' ' + w : w;
      if (ctx.measureText(test).width <= maxWidth) {
        line = test;
      } else {
        if (line) lines.push(line);
        line = w;
      }
    }
    if (line) lines.push(line);
    return lines;
  }
  function roundRectPath(ctx, x, y, w, h, r = 6) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function drawLabel(ctx, p, opts = {}) {
    const o = { ...DRAW_DEFAULTS, ...opts };
    const n = { ...DRAW_DEFAULTS.node, ...(opts.node || {}) };
    const r = p.radius ?? n.defaultRadius;

    // style knobs
    const font =
      o.labelFont ||
      '600 14px system-ui, -apple-system, Segoe UI, Roboto, Arial';
    const lineHeight = o.labelLineHeight ?? 18;
    const maxWidth = o.labelMaxWidth ?? 180;
    const gap = o.labelGap ?? 4; // distance from circle to label bg bottom
    const padX = o.labelPadX ?? 8;
    const padY = o.labelPadY ?? 6;
    const showBg = o.labelBg ?? false;

    ctx.font = font;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top'; // simplifies placement

    // wrap and measure
    const lines = wrapText(ctx, String(p.label || ''), maxWidth);
    const totalH = lines.length * lineHeight;

    // Anchor: bottom of bg is exactly gap above the circle
    const blockBottomY = p.y - r / 1.5 - gap;
    const blockTopY = blockBottomY - (totalH + padY * 2);

    // width for bg
    let widest = 0;
    for (const ln of lines)
      widest = Math.max(widest, ctx.measureText(ln).width);
    const bgW = widest + padX * 2;
    const bgH = totalH + padY * 2;
    const bgX = p.x - bgW / 2;

    if (showBg) {
      roundRectPath(ctx, bgX, blockTopY, bgW, bgH, 8);
      ctx.fillStyle = 'rgba(255,255,255,0.92)';
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.stroke();
    }

    // draw lines (top-aligned)
    lines.forEach((line, i) => {
      const y = blockTopY + padY + i * lineHeight;
      // subtle halo
      // ctx.lineWidth = 2;
      // ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      // ctx.strokeText(line, p.x, y);
      // ctx.fillStyle = o.labelColor || '#111';
      // ctx.fillText(line, p.x, y);
      ctx.fillStyle = o.labelColor || '#111';
      ctx.fillText(line, p.x, y);
    });
  }

  function isTooClose(x, y, points, except = null, minDistance = 50) {
    for (const p of points) {
      if (p === except) continue;
      const dx = p.x - x,
        dy = p.y - y;
      if (Math.hypot(dx, dy) < minDistance) return true;
    }
    return false;
  }

  function tryMove(point, x, y, points, draw, minDistance = 50, onReject) {
    if (!isTooClose(x, y, points, point, minDistance)) {
      point.x = x;
      point.y = y;
      if (typeof draw === 'function') draw();
      return true;
    }
    if (typeof onReject === 'function') onReject(point);
    return false;
  }

  // 🔴 shared flash helper
  function flash(point, draw, duration = 150) {
    point.flash = true;
    if (typeof draw === 'function') draw();
    setTimeout(() => {
      point.flash = false;
      if (typeof draw === 'function') draw();
    }, duration);
  }

  // ===== Drawing defaults (override per page if you want) =====
  const DRAW_DEFAULTS = {
    borderColor: '#000',
    borderWidth: 3,
    labelFont: '14px sans-serif',
    labelColor: '#111',
    node: {
      defaultRadius: 20,
      orange: '#5CA4A9', // '#f5a623',
      orangeSelected: '#8CC7CA', // '#ffd54f',
      grey: '#999999',
      greySelected: '#777777',
      flash: '#ff4d4d',
      stroke: '#000000',
      strokeWidth: 2,
    },
    edge: {
      positive: 'green',
      negative: 'red',
      minWidth: 1,
      maxWidth: 10, // = min + (strength/100)*(max-min)
    },
  };

  // ===== Utilities =====
  function setCanvasSize(canvas, width, height) {
    canvas.width = width;
    canvas.height = height;
  }

  function clearAndFrame(ctx, canvas, opts = {}) {
    const o = { ...DRAW_DEFAULTS, ...opts };
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = o.borderColor;
    ctx.lineWidth = o.borderWidth;
    ctx.strokeRect(0, 0, canvas.width, canvas.height);
  }

  // ===== Low-level primitives =====
  function drawEdges(ctx, edges = [], opts = {}) {
    const e = { ...DRAW_DEFAULTS.edge, ...(opts.edge || {}) };
    for (const edge of edges || []) {
      const w =
        e.minWidth + ((edge.strength ?? 50) / 100) * (e.maxWidth - e.minWidth);
      ctx.beginPath();
      ctx.moveTo(edge.from.x, edge.from.y);
      ctx.lineTo(edge.to.x, edge.to.y);
      ctx.strokeStyle = edge.polarity === 'negative' ? e.negative : e.positive;
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
    if (p.flash) {
      fill = n.flash;
    } else if (p === selectedPoint || p.selected) {
      fill =
        p.fixed || p.label === 'Meat Eating'
          ? n.greySelected
          : n.orangeSelected;
    } else if (p.fixed || p.label === 'Meat Eating') {
      fill = n.grey;
    } else {
      fill = n.orange;
    }

    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = n.stroke;
    ctx.lineWidth = n.strokeWidth;
    ctx.stroke();
  }

  // static/otreesurvey_app/canvasfunc.js
  (function (w) {
    'use strict';

    // --- helpers ---
    function wrapText(ctx, text, maxWidth) {
      const paras = String(text || '').split('\n');
      const lines = [];
      for (const para of paras) {
        const words = para.trim().split(/\s+/).filter(Boolean);
        if (words.length === 0) {
          lines.push('');
          continue;
        }
        let line = '';
        for (const w of words) {
          const test = line ? line + ' ' + w : w;
          if (ctx.measureText(test).width <= maxWidth) {
            line = test;
          } else {
            if (line) lines.push(line);
            line = w;
          }
        }
        if (line) lines.push(line);
      }
      return lines;
    }
    function roundRectPath(ctx, x, y, w, h, r = 6) {
      const rr = Math.min(r, w / 2, h / 2);
      ctx.beginPath();
      ctx.moveTo(x + rr, y);
      ctx.arcTo(x + w, y, x + w, y + h, rr);
      ctx.arcTo(x + w, y + h, x, y + h, rr);
      ctx.arcTo(x, y + h, x, y, rr);
      ctx.arcTo(x, y, x + w, y, rr);
      ctx.closePath();
    }

    // --- nicer labels ---
    // Centered-on-circle label (always drawn LAST by drawGraph)
    function drawLabel(ctx, p, opts = {}) {
      const o = { ...DRAW_DEFAULTS, ...opts };
      const n = { ...DRAW_DEFAULTS.node, ...(opts.node || {}) };
      const r = p.radius ?? n.defaultRadius;

      const pad = o.labelPad ?? 6; // inner padding to circle edge
      const baseFont =
        o.labelFont ||
        '600 14px system-ui, -apple-system, Segoe UI, Roboto, Arial';
      const lineHFactor = o.labelLineHeightFactor ?? 1.25;

      // start with base font
      ctx.font = baseFont;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // wrap to circle diameter (minus padding)
      const maxWidth = Math.max(10, 2 * r - 2 * pad);
      let lines = wrapText(ctx, p.label, maxWidth);

      // compute initial metrics
      const fontSizeMatch = /(\d+(?:\.\d+)?)px/.exec(ctx.font);
      let fontPx = fontSizeMatch ? parseFloat(fontSizeMatch[1]) : 14;
      let lineH = Math.round(fontPx * lineHFactor);

      // if the block is taller than the circle, scale down (to a floor)
      const maxHeight = 2 * r - 2 * pad;
      const blockH = lines.length * lineH;
      if (blockH > maxHeight) {
        const scale = Math.max(0.7, maxHeight / blockH); // don't go tiny: floor at 70%
        fontPx = Math.max(10, fontPx * scale);
        ctx.font = baseFont.replace(/(\d+(?:\.\d+)?)px/, `${fontPx}px`);
        lineH = Math.round(fontPx * lineHFactor);
        // re-wrap at the new width (font size affects word widths)
        lines = wrapText(ctx, p.label, maxWidth);
      }

      // vertical centering around p.y
      const totalH = lines.length * lineH;
      const y0 = p.y - (totalH - lineH) / 2;

      // draw with a subtle halo for readability
      for (let i = 0; i < lines.length; i++) {
        const y = y0 + i * lineH;
        ctx.lineWidth = 3;
        ctx.strokeStyle = 'rgba(255,255,255,0.9)';
        ctx.strokeText(lines[i], p.x, y);
        ctx.fillStyle = o.labelColor || '#111';
        ctx.fillText(lines[i], p.x, y);
      }
    }

    function isTooClose(x, y, points, except = null, minDistance = 50) {
      for (const p of points) {
        if (p === except) continue;
        const dx = p.x - x,
          dy = p.y - y;
        if (Math.hypot(dx, dy) < minDistance) return true;
      }
      return false;
    }

    function tryMove(point, x, y, points, draw, minDistance = 50, onReject) {
      if (!isTooClose(x, y, points, point, minDistance)) {
        point.x = x;
        point.y = y;
        if (typeof draw === 'function') draw();
        return true;
      }
      if (typeof onReject === 'function') onReject(point);
      return false;
    }

    // 🔴 shared flash helper
    function flash(point, draw, duration = 150) {
      point.flash = true;
      if (typeof draw === 'function') draw();
      setTimeout(() => {
        point.flash = false;
        if (typeof draw === 'function') draw();
      }, duration);
    }

    // ===== Utilities =====
    function setCanvasSize(canvas, width, height) {
      canvas.width = width;
      canvas.height = height;
    }

    function clearAndFrame(ctx, canvas, opts = {}) {
      const o = { ...DRAW_DEFAULTS, ...opts };
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = o.borderColor;
      ctx.lineWidth = o.borderWidth;
      ctx.strokeRect(0, 0, canvas.width, canvas.height);
    }

    // ===== Low-level primitives =====
    function drawEdges(ctx, edges = [], opts = {}) {
      const e = { ...DRAW_DEFAULTS.edge, ...(opts.edge || {}) };
      for (const edge of edges || []) {
        const w =
          e.minWidth +
          ((edge.strength ?? 50) / 100) * (e.maxWidth - e.minWidth);
        ctx.beginPath();
        ctx.moveTo(edge.from.x, edge.from.y);
        ctx.lineTo(edge.to.x, edge.to.y);
        ctx.strokeStyle =
          edge.polarity === 'negative' ? e.negative : e.positive;
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

      if (p.flash) {
        fill = n.flash;
      } else if (p === selectedPoint || p.selected) {
        // Selected state (either explicitly or via .selected flag)
        if (p.label === 'Meat Eating') {
          fill = '#777777'; // darker grey when grey node is selected
        } else {
          fill = 'yellow'; // yellow when orange node is selected
        }
      } else if (p.fixed || p.label === 'Meat Eating') {
        fill = '#999999'; // normal grey
      } else {
        fill = 'orange'; // normal orange
      }

      ctx.fillStyle = fill;
      ctx.fill();
      ctx.strokeStyle = n.stroke;
      ctx.lineWidth = n.strokeWidth;
      ctx.stroke();
    }

    function drawLabel(ctx, p, opts = {}) {
      const o = { ...DRAW_DEFAULTS, ...opts };
      ctx.font = o.labelFont;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      const lines = (p.label || '').split('\n');
      const lineHeight = 16;
      const totalHeight = lines.length * lineHeight;

      lines.forEach((line, i) => {
        const y = p.y - totalHeight / 2 + i * lineHeight;

        // halo
        // ctx.lineWidth = 4;
        // ctx.strokeStyle = 'rgba(255,255,255,0.85)';
        // ctx.strokeText(line, p.x, y);

        // fill
        // ctx.fillStyle = o.labelColor;
        // ctx.fillText(line, p.x, y);
        ctx.fillStyle = o.labelColor;
        ctx.fillText(line, p.x, y);
      });
    }

    // ===== One harmonized drawer =====
    function drawGraph(ctx, canvas, data, opts = {}) {
      clearAndFrame(ctx, canvas, opts);

      // 1. Edges first
      if (data.edges && data.edges.length) {
        drawEdges(ctx, data.edges, opts);
      }

      // 2. Draw all nodes
      (data.points || []).forEach((p) => {
        drawNode(ctx, p, data.selectedPoint, opts);
      });

      // 3. Draw all labels last (always on top)
      (data.points || []).forEach((p) => {
        drawLabel(ctx, p, opts);
      });
    }

    // ===== Safe hit-detection (with fallback radius) =====
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

    // Expose
    w.NoOverlap = Object.assign(w.NoOverlap || {}, {
      DRAW_DEFAULTS,
      setCanvasSize,
      clearAndFrame,
      drawEdges,
      drawNode,
      drawLabel,
      drawGraph,
      isTooClose,
      tryMove,
      flash,
      findPoint, // 👈 now available globally
    });
  })(window);

  // ===== One harmonized drawer =====
  function drawGraph(ctx, canvas, data, opts = {}) {
    clearAndFrame(ctx, canvas, opts);

    // 1. Edges first
    if (data.edges && data.edges.length) {
      drawEdges(ctx, data.edges, opts);
    }

    // 2. Draw all nodes
    (data.points || []).forEach((p) => {
      drawNode(ctx, p, data.selectedPoint, opts);
    });

    // 3. Draw all labels last (always on top)
    (data.points || []).forEach((p) => {
      drawLabel(ctx, p, opts);
    });
  }

  // ===== Safe hit-detection (with fallback radius) =====
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

  // Expose
  w.NoOverlap = Object.assign(w.NoOverlap || {}, {
    DRAW_DEFAULTS,
    setCanvasSize,
    clearAndFrame,
    drawEdges,
    drawNode,
    drawLabel,
    drawGraph,
    isTooClose,
    tryMove,
    flash,
    findPoint, // 👈 now available globally
  });
})(window);
