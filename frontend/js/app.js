/* ════════════════════════════════════════════════════════════
   APP.JS
   – Carga de CSV, entrenamiento, SSE, resultados, predicción
   – Depende de layers.js (layersState, nFeatures, updateDiagram)
════════════════════════════════════════════════════════════ */

var dataLoaded    = false;
var sseSource     = null;
var cfgMaxEpochs  = 300;
var cfgK          = 5;
var lcChart       = null;
var foldLogs      = {};   // { fold_num: [{epoch, error_train, error_val}, ...] }
var isMulticlass  = false;
var classNames    = [];

/* ════════════════════════════════════════════════════════════
   CSV UPLOAD
════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', function () {
  document.getElementById('csv-file').addEventListener('change', function (e) {
    var f = e.target.files[0];
    if (f) uploadCSV(f);
  });

  var dz = document.getElementById('drop-zone');
  dz.addEventListener('dragover', function (e) {
    e.preventDefault();
    dz.style.borderColor = 'var(--accent)';
  });
  dz.addEventListener('dragleave', function () {
    dz.style.borderColor = '';
  });
  dz.addEventListener('drop', function (e) {
    e.preventDefault();
    dz.style.borderColor = '';
    var f = e.dataTransfer.files[0];
    if (f && f.name.endsWith('.csv')) uploadCSV(f);
  });
});

function uploadCSV(file) {
  var fd = new FormData();
  fd.append('file', file);
  fd.append('has_header', document.getElementById('has-header').checked);
  fetch('/upload-csv', { method: 'POST', body: fd })
    .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
    .then(function (res) {
      if (res.ok) {
        isMulticlass = res.d.is_multiclass || false;
        classNames   = res.d.class_names   || [];

        var label = file.name + '  (' + res.d.num_samples + ' filas · ' + res.d.num_features + ' features)';
        if (isMulticlass) label += '  · Multiclase: ' + classNames.join(', ');
        document.getElementById('file-label').textContent = label;
        document.getElementById('drop-zone').classList.add('loaded');
        document.getElementById('train-btn').disabled = false;
        dataLoaded = true;
        nFeatures  = res.d.num_features;
        updateDiagram();

        // Mostrar preview del dataset
        if (res.d.preview) showDatasetPreview(res.d.preview, res.d.num_samples);

        // Auto-configurar salida para multiclase
        if (isMulticlass) {
          document.getElementById('neuronas-salida').value = res.d.n_classes;
          document.getElementById('activacion-salida').value = 'Softmax';
          updateDiagram();
          log('Multiclase detectado: ' + res.d.n_classes + ' clases (' + classNames.join(', ') + '). Salida auto-configurada: ' + res.d.n_classes + ' neuronas · Softmax');
        } else {
          log('CSV cargado: ' + res.d.num_samples + ' muestras, ' + res.d.num_features + ' features');
        }
      } else {
        alert('Error al cargar CSV: ' + (res.d.detail || 'desconocido'));
      }
    })
    .catch(function (err) { alert('Error de conexión: ' + err); });
}

/* ════════════════════════════════════════════════════════════
   DATASET PREVIEW
════════════════════════════════════════════════════════════ */
function showDatasetPreview(preview, totalRows) {
  var card   = document.getElementById('dataset-card');
  var info   = document.getElementById('dataset-info');
  var thead  = document.getElementById('dataset-thead');
  var tbody  = document.getElementById('dataset-tbody');

  var shown = preview.rows.length;
  info.textContent = totalRows + ' filas · ' + preview.columns.length + ' columnas' +
    (totalRows > shown ? '  (mostrando primeras ' + shown + ')' : '');

  thead.innerHTML = '<tr>' +
    preview.columns.map(function (c, i) {
      var isLast = i === preview.columns.length - 1;
      return '<th style="text-align:' + (isLast ? 'center' : 'right') + ';' +
             (isLast ? 'color:var(--green);' : '') + '">' + c + '</th>';
    }).join('') + '</tr>';

  tbody.innerHTML = preview.rows.map(function (row) {
    return '<tr>' + row.map(function (v, i) {
      var isLast = i === row.length - 1;
      return '<td style="' + (isLast ? 'color:var(--green);font-weight:700;text-align:center;' : '') + '">' + v + '</td>';
    }).join('') + '</tr>';
  }).join('');

  card.style.display = 'block';
}

/* ════════════════════════════════════════════════════════════
   LOG
════════════════════════════════════════════════════════════ */
function log(msg) {
  var box = document.getElementById('log-box');
  box.style.display = 'block';
  var d = document.createElement('div');
  d.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
  box.appendChild(d);
  box.scrollTop = box.scrollHeight;
}

/* ════════════════════════════════════════════════════════════
   TRAINING
════════════════════════════════════════════════════════════ */
function startTraining() {
  if (!dataLoaded) return;

  var n = +document.getElementById('num-capas').value;

  var capas_config = layersState.slice(0, n).map(function (l) {
    return { neuronas: +l.neuronas, activacion: l.activacion };
  });

  var config = {
    k:               +document.getElementById('k').value,
    eta:             +document.getElementById('eta').value,
    eps:             +document.getElementById('eps').value,
    capas_config:    capas_config,
    neuronas_salida: +document.getElementById('neuronas-salida').value,
    activacion_salida: document.getElementById('activacion-salida').value,
    max_epochs:      +document.getElementById('max-epochs').value,
    normalizar:       document.getElementById('normalizar').checked,
    paciencia:        +document.getElementById('paciencia').value,
  };

  cfgMaxEpochs = config.max_epochs;
  cfgK         = config.k;

  /* Reset UI */
  foldLogs = {};
  document.getElementById('prog-card').style.display       = 'block';
  document.getElementById('results-card').style.display    = 'none';
  document.getElementById('chart-card').style.display      = 'none';
  document.getElementById('predict-card').style.display    = 'none';
  document.getElementById('foldlogs-card').style.display   = 'none';
  document.getElementById('prog-bar').style.width       = '0%';
  document.getElementById('fold-status').textContent    = 'Iniciando...';
  document.getElementById('epoch-lbl').textContent      = 'Época —';
  document.getElementById('error-lbl').textContent      = '—';
  document.getElementById('train-btn').disabled         = true;
  var lb = document.getElementById('log-box');
  lb.innerHTML = ''; lb.style.display = 'none';

  /* SSE */
  if (sseSource) sseSource.close();
  sseSource = new EventSource('/progreso');
  sseSource.onmessage = function (e) { handleSSE(JSON.parse(e.data)); };
  sseSource.onerror   = function () {};

  /* POST */
  fetch('/entrenar', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
    .then(function (r) {
      if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail); });
      log('Entrenamiento iniciado...');
    })
    .catch(function (err) {
      alert('Error al iniciar: ' + err.message);
      document.getElementById('train-btn').disabled = false;
    });
}

/* ════════════════════════════════════════════════════════════
   SSE HANDLER
════════════════════════════════════════════════════════════ */
function handleSSE(ev) {
  if (ev.tipo === 'inicio') {
    log(ev.mensaje || 'Iniciando...');
  } else if (ev.tipo === 'datos') {
    log('Datos: ' + ev.n_samples + ' muestras · ' + ev.n_features + ' features');
  } else if (ev.tipo === 'validacion_inicio') {
    log('Iniciando ' + ev.k + '-Fold Cross Validation');
  } else if (ev.tipo === 'epoca') {
    /* Acumula datos para los logs por fold */
    if (!foldLogs[ev.fold]) foldLogs[ev.fold] = [];
    foldLogs[ev.fold].push({
      epoch:       ev.epoca,
      error_train: ev.error_train,
      error_val:   ev.error_val,
    });

    var frac = (ev.fold - 1) / cfgK + ev.epoca / (cfgMaxEpochs * cfgK);
    document.getElementById('prog-bar').style.width =
      Math.min(frac * 100, 98) + '%';
    document.getElementById('fold-status').textContent =
      'Fold ' + ev.fold + ' / ' + cfgK + '  ·  Época ' + ev.epoca + ' / ' + cfgMaxEpochs;
    document.getElementById('epoch-lbl').textContent =
      'Fold ' + ev.fold + ' · Época ' + ev.epoca;
    document.getElementById('error-lbl').textContent =
      'Train ' + ev.error_train.toFixed(4) + '  |  Val ' + ev.error_val.toFixed(4);
  } else if (ev.tipo === 'fin') {
    document.getElementById('prog-bar').style.width    = '100%';
    document.getElementById('fold-status').textContent =
      'Completado ✓  —  Mejor fold: ' + ev.mejor_fold;
    log('Entrenamiento completo. Mejor fold: ' + ev.mejor_fold);
    displayResults(ev.resultado);
    document.getElementById('train-btn').disabled = false;
  } else if (ev.tipo === 'error') {
    log('ERROR: ' + ev.mensaje);
    document.getElementById('fold-status').textContent = 'Error durante el entrenamiento.';
    document.getElementById('train-btn').disabled = false;
  } else if (ev.tipo === 'fin_stream') {
    if (sseSource) { sseSource.close(); sseSource = null; }
  }
}

/* ════════════════════════════════════════════════════════════
   RESULTS
════════════════════════════════════════════════════════════ */
function displayResults(result) {
  /* Summary stats */
  var sg = document.getElementById('summary-grid');
  sg.innerHTML = '';
  var stats = [
    { label: 'K Folds',           value: result.k },
    { label: 'Muestras',          value: result.total_samples },
    { label: 'Error Train Prom.', value: result.error_train_promedio.toFixed(4) },
    { label: '± Std Train',       value: result.std_train.toFixed(4) },
    { label: 'Error Val Prom.',   value: result.error_test_promedio.toFixed(4) },
    { label: '± Std Val',         value: result.std_test.toFixed(4) },
    { label: 'Error Total Pond.', value: result.error_total_promedio.toFixed(4) },
    { label: '± Std Total',       value: result.std_total.toFixed(4) },
  ];
  if (result.is_multiclass) {
    stats.push({ label: 'Accuracy Train', value: (result.accuracy_train_prom * 100).toFixed(1) + '%', green: true });
    stats.push({ label: 'Accuracy Val',   value: (result.accuracy_test_prom  * 100).toFixed(1) + '%', green: true });
  }
  stats.push({ label: 'Mejor Fold', value: result.mejor_fold, green: true });
  stats.forEach(function (s) {
    sg.innerHTML +=
      '<div class="stat">' +
        '<div class="stat-label">' + s.label + '</div>' +
        '<div class="stat-value ' + (s.green ? 'green' : '') + '">' + s.value + '</div>' +
      '</div>';
  });

  /* Fold cards */
  var fg = document.getElementById('folds-grid');
  fg.innerHTML = '';
  result.folds.forEach(function (fold) {
    var best = fold.fold === result.mejor_fold;
    var accHtml = result.is_multiclass
      ? '<div class="kv"><span>Accuracy Train</span><span>' + (fold.accuracy_train * 100).toFixed(1) + '%</span></div>' +
        '<div class="kv"><span>Accuracy Val</span><span>'   + (fold.accuracy_test  * 100).toFixed(1) + '%</span></div>'
      : '';
    fg.innerHTML +=
      '<div class="fold-card ' + (best ? 'best' : '') + '">' +
        '<div class="fold-title">Fold ' + fold.fold +
          (best ? ' <span class="badge">Mejor</span>' : '') + '</div>' +
        '<div class="kv"><span>Error Train</span><span>'       + fold.error_train.toFixed(4)  + '</span></div>' +
        '<div class="kv"><span>Error Val</span><span>'         + fold.error_test.toFixed(4)   + '</span></div>' +
        '<div class="kv"><span>Error Total Pond.</span><span>' + fold.error_total.toFixed(4)  + '</span></div>' +
        accHtml +
        '<div class="kv"><span>Época conv.</span><span>'       + fold.epoca_convergencia       + '</span></div>' +
        '<div class="kv"><span>Épocas tot.</span><span>'       + fold.epocas                   + '</span></div>' +
      '</div>';
  });

  /* Chart */
  renderChart(result.mejor_fold_historial_train, result.mejor_fold_historial_test);

  /* Fold logs */
  displayFoldLogs(result);

  /* Comparaciones */
  loadComparaciones();

  document.getElementById('results-card').style.display  = 'block';
  document.getElementById('chart-card').style.display    = 'block';
  document.getElementById('predict-card').style.display  = 'block';
}

/* ════════════════════════════════════════════════════════════
   CHART — curvas de aprendizaje
════════════════════════════════════════════════════════════ */
function renderChart(trainData, valData) {
  var ctx = document.getElementById('lc-chart').getContext('2d');
  if (lcChart) lcChart.destroy();
  lcChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: trainData.map(function (_, i) { return i + 1; }),
      datasets: [
        {
          label: 'Error Train',
          data: trainData,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37,99,235,.07)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35,
          fill: true,
        },
        {
          label: 'Error Val',
          data: valData,
          borderColor: '#dc2626',
          backgroundColor: 'rgba(220,38,38,.05)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.35,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: function (ctx) {
              return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(5);
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: 'Época',       font: { size: 11 } }, ticks: { maxTicksLimit: 10 } },
        y: { title: { display: true, text: 'Error (L2)', font: { size: 11 } } },
      },
    },
  });
}

/* ════════════════════════════════════════════════════════════
   PREDICT
════════════════════════════════════════════════════════════ */
function runPredict() {
  var raw = document.getElementById('predict-input').value.trim();
  if (!raw) return;

  var datos = raw.split('\n').filter(function (l) { return l.trim(); });

  var normalizar = document.getElementById('normalizar').checked;

  fetch('/predecir', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ datos: datos, normalizar: normalizar }),
  })
    .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
    .then(function (res) {
      if (!res.ok) throw new Error(res.d.detail);
      var el = document.getElementById('pred-result');
      el.style.display = 'block';
      if (res.d.probabilidades) {
        // Multiclase: mostrar clase predicha + probabilidades
        el.textContent = 'Fold usado: ' + res.d.mejor_fold + '\n\n' +
          res.d.predicciones.map(function (clase, i) {
            var probs = res.d.probabilidades[i];
            var probStr = res.d.clases.map(function (c, j) {
              return c + ': ' + (probs[j] * 100).toFixed(1) + '%';
            }).join('  |  ');
            return 'Muestra ' + (i + 1) + ': ' + clase + '\n  ' + probStr;
          }).join('\n');
      } else {
        el.textContent =
          'Fold usado: ' + res.d.mejor_fold + '\n' +
          res.d.predicciones.map(function (p, i) {
            return 'Muestra ' + (i + 1) + ': ' + (typeof p === 'number' ? p.toFixed(6) : p);
          }).join('\n');
      }
    })
    .catch(function (err) { alert('Error en predicción: ' + err.message); });
}

/* ════════════════════════════════════════════════════════════
   COMPARACIÓN DE MODELOS
════════════════════════════════════════════════════════════ */
function loadComparaciones() {
  fetch('/comparaciones')
    .then(function (r) { return r.json(); })
    .then(function (data) {
      if (!data || data.length === 0) return;

      var hasAcc   = data.some(function (r) { return r.accuracy !== undefined; });
      var thead    = document.getElementById('compare-thead');
      var tbody    = document.getElementById('compare-tbody');

      thead.innerHTML =
        '<tr>' +
          '<th style="text-align:center;">#</th>' +
          '<th style="text-align:left;">Arquitectura</th>' +
          '<th>K</th>' +
          '<th>Error Train</th>' +
          '<th>Error Val</th>' +
          '<th>Error Total</th>' +
          (hasAcc ? '<th>Accuracy Val</th>' : '') +
          '<th>Mejor Fold</th>' +
          '<th>Tipo</th>' +
        '</tr>';

      // Ordenar por error_total ascendente
      var sorted = data.slice().sort(function (a, b) { return a.error_total - b.error_total; });

      tbody.innerHTML = sorted.map(function (r, idx) {
        var isBest = idx === 0;
        return '<tr' + (isBest ? ' style="background:var(--accent-lt);font-weight:700;"' : '') + '>' +
          '<td style="text-align:center;">' + r.run + (isBest ? ' ★' : '') + '</td>' +
          '<td style="text-align:left;font-family:monospace;font-size:.72rem;">' + r.arquitectura + '</td>' +
          '<td>' + r.k + '</td>' +
          '<td>' + r.error_train.toFixed(4) + '</td>' +
          '<td>' + r.error_val.toFixed(4) + '</td>' +
          '<td>' + r.error_total.toFixed(4) + '</td>' +
          (hasAcc ? '<td>' + (r.accuracy !== undefined ? (r.accuracy * 100).toFixed(1) + '%' : '—') + '</td>' : '') +
          '<td>' + r.mejor_fold + '</td>' +
          '<td>' + r.tipo + '</td>' +
        '</tr>';
      }).join('');

      document.getElementById('compare-card').style.display = 'block';
    });
}

function clearComparaciones() {
  fetch('/comparaciones', { method: 'DELETE' })
    .then(function () {
      document.getElementById('compare-card').style.display = 'none';
      document.getElementById('compare-tbody').innerHTML = '';
    });
}

/* ════════════════════════════════════════════════════════════
   DESCARGA DEL DIAGRAMA
════════════════════════════════════════════════════════════ */
function downloadDiagram() {
  var canvas = document.getElementById('net-diagram');
  var link   = document.createElement('a');
  link.download = 'diagrama_red_neuronal.png';
  link.href     = canvas.toDataURL('image/png');
  link.click();
}

/* ════════════════════════════════════════════════════════════
   LOGS POR FOLD
════════════════════════════════════════════════════════════ */
function displayFoldLogs(result) {
  var container = document.getElementById('foldlogs-container');
  container.innerHTML = '';

  result.folds.forEach(function (fold) {
    var isBest    = fold.fold === result.mejor_fold;
    var convEpoch = fold.epoca_convergencia;
    var rows      = foldLogs[fold.fold] || [];

    /* ── details/summary (colapsable) ── */
    var details = document.createElement('details');
    details.className = 'fold-log-details';
    if (isBest) details.open = true;  // mejor fold abierto por defecto

    var summary = document.createElement('summary');
    summary.innerHTML =
      'Fold ' + fold.fold +
      (isBest ? '&nbsp;<span class="badge">Mejor</span>' : '') +
      '<span style="margin-left:auto;font-weight:400;font-size:.75rem;color:inherit;">' +
        'N train: ' + fold.n_train +
        ' &nbsp;|&nbsp; N val: '  + fold.n_test +
        ' &nbsp;|&nbsp; Train: '  + fold.error_train.toFixed(4) +
        ' &nbsp;|&nbsp; Val: '    + fold.error_test.toFixed(4)  +
        ' &nbsp;|&nbsp; Total: '  + fold.error_total.toFixed(4) +
        ' &nbsp;|&nbsp; Conv.: '  + convEpoch +
      '</span>';
    details.appendChild(summary);

    /* ── tabla de épocas ── */
    if (rows.length === 0) {
      var msg = document.createElement('p');
      msg.textContent = 'Sin datos de época recolectados.';
      msg.style.cssText = 'padding:8px 12px;font-size:.78rem;color:var(--gray-400);';
      details.appendChild(msg);
    } else {
      var wrap = document.createElement('div');
      wrap.className = 'fold-log-scroll';

      var table  = document.createElement('table');
      table.className = 'log-table';

      var nTrain  = fold.n_train;
      var nTest   = fold.n_test;
      var nTot    = nTrain + nTest;

      var thead = document.createElement('thead');
      thead.innerHTML =
        '<tr>' +
          '<th>Época</th>' +
          '<th>Error Train</th>' +
          '<th>Error Val</th>' +
          '<th>Error Total</th>' +
          '<th>|Ee − Ev|</th>' +
        '</tr>';
      table.appendChild(thead);

      var tbody = document.createElement('tbody');
      rows.forEach(function (row) {
        var isConv    = row.epoch === convEpoch;
        var diff      = Math.abs(row.error_train - row.error_val);
        var errTotal  = (row.error_train * nTrain + row.error_val * nTest) / nTot;
        var tr        = document.createElement('tr');
        if (isConv) tr.className = 'conv-row';
        tr.innerHTML =
          '<td>' + row.epoch + (isConv ? ' ★' : '') + '</td>' +
          '<td>' + row.error_train.toFixed(5) + '</td>' +
          '<td>' + row.error_val.toFixed(5)   + '</td>' +
          '<td>' + errTotal.toFixed(5)         + '</td>' +
          '<td>' + diff.toFixed(5)             + '</td>';
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      details.appendChild(wrap);
    }

    container.appendChild(details);
  });

  document.getElementById('foldlogs-card').style.display = 'block';
}
