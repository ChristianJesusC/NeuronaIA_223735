/* ════════════════════════════════════════════════════════════
   LAYERS.JS
   – Gestiona la tabla de capas ocultas + capa de salida
   – Dibuja el diagrama de red en el canvas #net-diagram
════════════════════════════════════════════════════════════ */

/* ── Estado persistente de capas ─────────────────────────── */
var layersState = [
  { neuronas: 64, activacion: 'ReLU' },
  { neuronas: 32, activacion: 'ReLU' },
];

/* nFeatures se actualiza desde app.js cuando se carga el CSV */
var nFeatures = 0;

var ACTIVACIONES = ['ReLU', 'Lineal', 'Sigmoide', 'Binaria'];

/* ── Tabla de capas ocultas ──────────────────────────────── */
function setNumCapas(n) {
  n = Math.max(1, Math.min(12, n));
  while (layersState.length < n) layersState.push({ neuronas: 32, activacion: 'ReLU' });
  renderLayers(n);
}

function renderLayers(n) {
  var tbody = document.getElementById('layers-body');
  tbody.innerHTML = '';

  for (var i = 0; i < n; i++) {
    if (!layersState[i]) layersState[i] = { neuronas: 32, activacion: 'ReLU' };
    var layer = layersState[i];

    var tr = document.createElement('tr');

    // Label
    var tdL = document.createElement('td');
    tdL.textContent = 'Oculta ' + (i + 1);
    tr.appendChild(tdL);

    // Neurons input
    var tdN = document.createElement('td');
    var inp = document.createElement('input');
    inp.type = 'number';
    inp.value = layer.neuronas;
    inp.min = 1; inp.max = 1024;
    inp.dataset.idx = i;
    inp.addEventListener('input', function () {
      layersState[+this.dataset.idx].neuronas = Math.max(1, +this.value || 1);
      updateDiagram();
    });
    tdN.appendChild(inp);
    tr.appendChild(tdN);

    // Activation select
    var tdA = document.createElement('td');
    var sel = document.createElement('select');
    sel.dataset.idx = i;
    ACTIVACIONES.forEach(function (a) {
      var opt = document.createElement('option');
      opt.value = a; opt.textContent = a;
      if (a === layer.activacion) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener('change', function () {
      layersState[+this.dataset.idx].activacion = this.value;
      updateDiagram();
    });
    tdA.appendChild(sel);
    tr.appendChild(tdA);

    tbody.appendChild(tr);
  }

  updateDiagram();
}

/* ── Output-layer controls (footer row) ──────────────────── */
function initOutputControls() {
  var neuronasOut = document.getElementById('neuronas-salida');
  var activOut    = document.getElementById('activacion-salida');
  if (neuronasOut) neuronasOut.addEventListener('input',  updateDiagram);
  if (activOut)    activOut.addEventListener('change', updateDiagram);
}

/* ════════════════════════════════════════════════════════════
   DIAGRAMA DE RED (Canvas)
════════════════════════════════════════════════════════════ */
var DIAG = {
  MAX_N:    5,    // neuronas máximas visibles por columna
  R:        13,   // radio de cada neurona
  HEADER_H: 62,   // alto reservado para etiquetas superiores
  FOOTER_H: 44,   // alto reservado para fórmula de pesos
};

/* Paleta por tipo de capa */
var LAYER_COLORS = {
  input:  { fill: '#dbeafe', stroke: '#2563eb', label: '#1d4ed8' },
  hidden: { fill: '#ede9fe', stroke: '#7c3aed', label: '#5b21b6' },
  output: { fill: '#d1fae5', stroke: '#059669', label: '#047857' },
};

function updateDiagram() {
  var canvas = document.getElementById('net-diagram');
  if (!canvas) return;

  var MIN_COL_W   = 140; // píxeles mínimos por columna
  var n           = +document.getElementById('num-capas').value;
  var totalCols   = 1 + n + 1; // entrada + ocultas + salida
  var contentW    = totalCols * MIN_COL_W;
  /* El canvas ocupa al menos el ancho del contenedor scroll;
     si hay más capas, crece (y el wrapper activa el scroll) */
  var wrapW = canvas.parentElement.clientWidth;
  canvas.width  = Math.max(contentW, wrapW > 20 ? wrapW : 400);
  canvas.height = 300;

  var n       = +document.getElementById('num-capas').value;
  var nSalida = +(document.getElementById('neuronas-salida') || {}).value || 1;
  var actSal  = (document.getElementById('activacion-salida') || {}).value || 'Lineal';
  var feats   = nFeatures > 0 ? nFeatures : 3; // 3 = valor ilustrativo

  /* Construye la descripción de todas las capas */
  var allLayers = [];

  allLayers.push({
    label:    'ENTRADA',
    sublabel: nFeatures > 0 ? ('X\u1d50\u02E3' + nFeatures) : 'X\u1d50\u02E3n',
    count:    feats,
    fa:       null,
    type:     'input',
  });

  for (var i = 0; i < n; i++) {
    var sl = layersState[i] || { neuronas: 32, activacion: 'ReLU' };
    allLayers.push({
      label:    'OCULTA ' + (i + 1),
      sublabel: null,
      count:    sl.neuronas,
      fa:       sl.activacion,
      type:     'hidden',
    });
  }

  allLayers.push({
    label:    'SALIDA',
    sublabel: null,
    count:    nSalida,
    fa:       actSal,
    type:     'output',
  });

  drawDiagram(canvas, allLayers, feats);
}

function drawDiagram(canvas, allLayers, n_features) {
  var ctx   = canvas.getContext('2d');
  var W     = canvas.width;
  var H     = canvas.height;
  var drawH = H - DIAG.HEADER_H - DIAG.FOOTER_H;
  var nCols = allLayers.length;
  var colW  = W / nCols;

  ctx.clearRect(0, 0, W, H);

  /* Calcula posiciones de neuronas por columna */
  var cols = allLayers.map(function (layer, ci) {
    var display  = Math.min(layer.count, DIAG.MAX_N);
    var hasMore  = layer.count > DIAG.MAX_N;
    var spacing  = drawH / (display + 1);
    var ys = [];
    for (var j = 0; j < display; j++) ys.push(DIAG.HEADER_H + spacing * (j + 1));
    return Object.assign({}, layer, {
      x: colW * ci + colW / 2,
      ys: ys,
      display: display,
      hasMore: hasMore,
    });
  });

  /* ── Separadores verticales de sección ────────────────── */
  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth   = 1;
  ctx.setLineDash([4, 3]);
  for (var ci = 1; ci < cols.length; ci++) {
    var sepX = cols[ci].x - colW / 2;
    ctx.beginPath();
    ctx.moveTo(sepX, 8);
    ctx.lineTo(sepX, H - DIAG.FOOTER_H - 4);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  /* ── Conexiones ───────────────────────────────────────── */
  ctx.globalAlpha = 0.5;
  for (var ci = 0; ci < cols.length - 1; ci++) {
    var from = cols[ci];
    var to   = cols[ci + 1];
    /* Color de la conexión = tono del destino */
    ctx.strokeStyle = LAYER_COLORS[to.type].stroke;
    ctx.lineWidth   = 0.85;
    from.ys.forEach(function (y1) {
      to.ys.forEach(function (y2) {
        ctx.beginPath();
        ctx.moveTo(from.x + DIAG.R, y1);
        ctx.lineTo(to.x - DIAG.R,   y2);
        ctx.stroke();
      });
    });
  }
  ctx.globalAlpha = 1;

  /* ── Neuronas ─────────────────────────────────────────── */
  cols.forEach(function (col) {
    var c = LAYER_COLORS[col.type];
    col.ys.forEach(function (y) {
      ctx.beginPath();
      ctx.arc(col.x, y, DIAG.R, 0, Math.PI * 2);
      ctx.fillStyle   = c.fill;
      ctx.strokeStyle = c.stroke;
      ctx.lineWidth   = 2;
      ctx.fill();
      ctx.stroke();
    });
    if (col.hasMore) {
      ctx.fillStyle  = '#9aa3b0';
      ctx.font       = '16px system-ui';
      ctx.textAlign  = 'center';
      ctx.fillText('\u22EE', col.x, col.ys[col.ys.length - 1] + DIAG.R + 16);
    }
  });

  /* ── Etiquetas superiores ─────────────────────────────── */
  cols.forEach(function (col) {
    var c = LAYER_COLORS[col.type];
    ctx.textAlign = 'center';
    var y = 14;

    ctx.font      = 'bold 11px system-ui';
    ctx.fillStyle = c.label;
    ctx.fillText(col.label, col.x, y); y += 13;

    if (col.sublabel) {
      ctx.font      = '9px monospace';
      ctx.fillStyle = '#9aa3b0';
      ctx.fillText(col.sublabel, col.x, y); y += 12;
    }
    if (col.fa) {
      ctx.font      = 'italic 9px system-ui';
      ctx.fillStyle = '#6b7280';
      ctx.fillText('FA: ' + col.fa, col.x, y); y += 12;
    }
    ctx.font      = '9px system-ui';
    ctx.fillStyle = '#9aa3b0';
    ctx.fillText(col.count + ' nrn' + (col.count !== 1 ? 's' : ''), col.x, y);
  });

  /* ── Fórmula de pesos (footer) ────────────────────────── */
  var total = 0;
  var parts = [];
  var prevN = n_features;

  for (var li = 1; li < allLayers.length; li++) {
    var currN = allLayers[li].count;
    total += prevN * currN + currN;
    parts.push(currN + '\u00D7' + prevN + '+' + currN);
    prevN = currN;
  }

  /* Línea separadora del footer */
  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(10, H - DIAG.FOOTER_H);
  ctx.lineTo(W - 10, H - DIAG.FOOTER_H);
  ctx.stroke();

  ctx.fillStyle = '#374151';
  ctx.font      = 'bold 11px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('Total pesos: ' + total, W / 2, H - 26);

  ctx.fillStyle = '#9aa3b0';
  ctx.font      = '9px system-ui';
  var fmtStr = parts.join(' + ');
  ctx.fillText(
    ctx.measureText(fmtStr).width < W - 24 ? fmtStr : parts.length + ' grupos de capas',
    W / 2, H - 10
  );

  /* Nota si CSV no cargado */
  if (nFeatures === 0) {
    ctx.fillStyle = '#f59e0b';
    ctx.font      = 'italic 9px system-ui';
    ctx.textAlign = 'left';
    ctx.fillText('* Entrada ilustrativa (CSV no cargado)', 10, H - DIAG.FOOTER_H - 4);
  }
}

/* ── Init ────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', function () {
  document.getElementById('num-capas').value = layersState.length;
  renderLayers(layersState.length);
  initOutputControls();
  window.addEventListener('resize', updateDiagram);
});
