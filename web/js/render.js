// Drawing the shot tracer on a 2D canvas.

export const STYLES = ['Glow', 'Solid', 'Comet'];

function rgba(hex, a) {
  const h = hex.replace('#', '');
  return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`;
}

function lighten(hex, amount = 0.6) {
  const h = hex.replace('#', '');
  const c = [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  return `rgb(${c.map((v) => Math.round(v + (255 - v) * amount)).join(',')})`;
}

/**
 * Draw the first `count` points of `path` (video pixel coords) scaled by `scale`.
 * style: {color, style, thickness (px at 1080p), marker}; height: canvas height in px.
 */
export function drawTracer(ctx, path, count, style, scale, height, headActive = true) {
  if (!path || count <= 0) return;
  const th = Math.max(1.5, (style.thickness * height) / 1080);
  const X = (i) => path.xs[i] * scale;
  const Y = (i) => path.ys[i] * scale;
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const line = () => {
    ctx.beginPath();
    ctx.moveTo(X(0), Y(0));
    for (let i = 1; i < count; i++) ctx.lineTo(X(i), Y(i));
  };
  if (count >= 2) {
    if (style.style === 'Solid') {
      line();
      ctx.strokeStyle = 'rgba(15,15,15,0.85)'; ctx.lineWidth = th + 2.5; ctx.stroke();
      ctx.strokeStyle = style.color; ctx.lineWidth = th; ctx.stroke();
    } else if (style.style === 'Comet') {
      // tail fades in and thickens towards the ball
      for (let i = 1; i < count; i++) {
        const f = i / (count - 1);
        ctx.beginPath();
        ctx.moveTo(X(i - 1), Y(i - 1));
        ctx.lineTo(X(i), Y(i));
        ctx.strokeStyle = rgba(style.color, f ** 1.5);
        ctx.lineWidth = th * (0.35 + 0.65 * f);
        ctx.stroke();
      }
    } else {
      line();
      for (const [w, a] of [[4.2, 0.1], [2.6, 0.22], [1.7, 0.45]]) {
        ctx.strokeStyle = rgba(style.color, a); ctx.lineWidth = th * w; ctx.stroke();
      }
      ctx.strokeStyle = style.color; ctx.lineWidth = th; ctx.stroke();
      ctx.strokeStyle = lighten(style.color); ctx.lineWidth = Math.max(1, th / 3); ctx.stroke();
    }
  }
  if (style.marker && headActive) {
    const x = X(count - 1), y = Y(count - 1), r = Math.max(2, th * 1.1);
    const g = ctx.createRadialGradient(x, y, 0, x, y, r * 3);
    g.addColorStop(0, rgba(style.color, 0.75));
    g.addColorStop(1, rgba(style.color, 0));
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r * 3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
}

export function drawMarker(ctx, x, y, label, color = '#36e27b') {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x - 17, y); ctx.lineTo(x - 6, y); ctx.moveTo(x + 6, y); ctx.lineTo(x + 17, y); ctx.stroke();
  if (label) {
    ctx.font = '600 13px system-ui, sans-serif';
    ctx.fillStyle = '#fff';
    ctx.shadowColor = 'rgba(0,0,0,.8)'; ctx.shadowBlur = 4;
    ctx.fillText(label, x + 15, y - 12);
  }
  ctx.restore();
}

export function drawPoints(ctx, xy, scale, color, r = 2.5, ring = false) {
  ctx.save();
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.5;
  for (let i = 0; i < xy.length; i += 2) {
    ctx.beginPath();
    ctx.arc(xy[i] * scale, xy[i + 1] * scale, r, 0, Math.PI * 2);
    if (ring) ctx.stroke(); else ctx.fill();
  }
  ctx.restore();
}
