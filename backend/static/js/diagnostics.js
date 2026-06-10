/**
 * Shared gloves diagnostics modal + live packet viewer.
 * Used by both collection.html and index.html.
 */
(function (global) {
  'use strict';

  const API_BASE = global.location ? global.location.origin : '';
  const WS_PROTO = global.location && global.location.protocol === 'https:' ? 'wss:' : 'ws:';

  let _ws = null;
  let _wsToken = 0;
  let _shouldReconnect = false;
  let _paused = false;
  let _handFilter = 0; // 0=both, 1=right, 2=left
  let _endpoint = global.location && global.location.pathname.startsWith('/collection')
    ? 'collect'
    : 'predict';
  let _packetLog = [];
  const MAX_LOG = 50;

  function _diagWsUrl() {
    const host = global.location ? global.location.host : 'localhost:8000';
    return `${WS_PROTO}//${host}/ws/diagnostics?endpoint=${encodeURIComponent(_endpoint)}`;
  }

  function _endpointPath() {
    return _endpoint === 'predict' ? '/ws/predict' : '/ws/collect';
  }

  function _statsUrl() {
    return _endpoint === 'predict'
      ? `${API_BASE}/api/predict/diagnostics`
      : `${API_BASE}/api/collection/diagnostics`;
  }

  function _injectModal() {
    if (document.getElementById('diag-modal')) return;

    const el = document.createElement('div');
    el.id = 'diag-modal';
    el.style.cssText = `
      display:none;position:fixed;inset:0;z-index:9999;
      background:rgba(0,0,0,.65);backdrop-filter:blur(2px);
      align-items:center;justify-content:center;
    `;
    el.innerHTML = `
      <div id="diag-panel" style="
        background:#1e293b;border:1px solid #334155;border-radius:.9rem;
        width:min(860px,96vw);max-height:90vh;overflow-y:auto;
        padding:1.5rem;font-family:system-ui,-apple-system,sans-serif;
        color:#e2e8f0;position:relative;
      ">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.1rem">
          <h2 style="font-size:1.05rem;font-weight:700;color:#3b82f6;margin:0">
            Gloves Diagnostics
          </h2>
          <button onclick="DiagnosticsModal.close()" style="
            background:none;border:none;color:#94a3b8;font-size:1.4rem;cursor:pointer;
            padding:.1rem .5rem;border-radius:.3rem;
          ">x</button>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem">
          <div class="diag-card">
            <div class="diag-label">Endpoint</div>
            <select id="diag-endpoint-select" onchange="DiagnosticsModal._setEndpoint(this.value)" style="
              width:100%;background:#0f172a;border:1px solid #334155;color:#e2e8f0;
              border-radius:.35rem;padding:.35rem .5rem;font-size:.82rem;margin-bottom:.65rem;
            ">
              <option value="predict">Prediction (/ws/predict)</option>
              <option value="collect">Data collection (/ws/collect)</option>
            </select>
            <div class="diag-label">WebSocket URL (ESP32 firmware)</div>
            <code id="diag-ws-url" style="color:#3b82f6;font-size:.82rem;word-break:break-all"></code>
            <div style="margin-top:.4rem">
              <span class="diag-label">Server LAN IPs:</span>
              <span id="diag-lan-ips" style="font-size:.82rem;color:#e2e8f0"></span>
              <button onclick="DiagnosticsModal._copyLanIp()" style="
                margin-left:.5rem;background:#334155;border:none;color:#e2e8f0;
                border-radius:.3rem;padding:.15rem .5rem;font-size:.75rem;cursor:pointer;
              ">Copy first</button>
            </div>
          </div>
          <div class="diag-card">
            <div class="diag-label">Active Session</div>
            <div id="diag-active-session" style="font-size:.85rem"></div>
            <div class="diag-label" style="margin-top:.5rem">Server Time</div>
            <div id="diag-server-time" style="font-size:.82rem;color:#94a3b8"></div>
          </div>
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem">
          <div class="diag-card">
            <div class="diag-label">Right Glove (hand_id=1)</div>
            <div style="display:flex;align-items:center;gap:.75rem;margin-top:.3rem">
              <div id="diag-hz-right" class="diag-hz">- Hz</div>
              <div id="diag-age-right" style="font-size:.78rem;color:#94a3b8"></div>
            </div>
            <div id="diag-schema-right" style="font-size:.76rem;margin-top:.4rem;color:#94a3b8"></div>
          </div>
          <div class="diag-card">
            <div class="diag-label">Left Glove (hand_id=2)</div>
            <div style="display:flex;align-items:center;gap:.75rem;margin-top:.3rem">
              <div id="diag-hz-left" class="diag-hz">- Hz</div>
              <div id="diag-age-left" style="font-size:.78rem;color:#94a3b8"></div>
            </div>
            <div id="diag-schema-left" style="font-size:.76rem;margin-top:.4rem;color:#94a3b8"></div>
          </div>
        </div>

        <details style="margin-bottom:1rem">
          <summary style="cursor:pointer;font-size:.85rem;font-weight:600;color:#94a3b8;padding:.4rem 0">
            Troubleshooting checklist
          </summary>
          <ol style="font-size:.82rem;color:#cbd5e1;line-height:1.7;margin:.6rem 0 0 1.2rem">
            <li>ESP32 and this PC must be on the same Wi-Fi network.</li>
            <li>Start uvicorn with <code>--host 0.0.0.0</code>.</li>
            <li>Set ESP32 <code>WS_HOST</code> to one LAN IP above and choose the endpoint shown here.</li>
            <li>Check Windows Firewall allows inbound TCP port 8000.</li>
            <li>Confirm both gloves show WebSocket connected in their serial monitors.</li>
            <li>If observed Hz drops below 25, check USB power or battery voltage.</li>
            <li>Schema errors usually mean firmware field names or array lengths do not match the backend.</li>
          </ol>
        </details>

        <div style="margin-bottom:.6rem;display:flex;align-items:center;gap:.6rem;flex-wrap:wrap">
          <span style="font-size:.85rem;font-weight:600;color:#94a3b8">Live Packet Viewer</span>
          <button id="diag-pause-btn" onclick="DiagnosticsModal._togglePause()" style="
            background:#334155;border:none;color:#e2e8f0;border-radius:.35rem;
            padding:.2rem .65rem;font-size:.78rem;cursor:pointer;
          ">Pause</button>
          <select id="diag-hand-filter" onchange="DiagnosticsModal._setFilter(this.value)" style="
            background:#0f172a;border:1px solid #334155;color:#e2e8f0;border-radius:.35rem;
            padding:.2rem .5rem;font-size:.78rem;
          ">
            <option value="0">Both hands</option>
            <option value="1">Right only</option>
            <option value="2">Left only</option>
          </select>
          <button onclick="DiagnosticsModal._copyLog()" style="
            background:#334155;border:none;color:#e2e8f0;border-radius:.35rem;
            padding:.2rem .65rem;font-size:.78rem;cursor:pointer;margin-left:auto;
          ">Copy log</button>
        </div>
        <pre id="diag-packet-log" style="
          background:#0f172a;border:1px solid #334155;border-radius:.5rem;
          padding:.75rem;font-size:.72rem;max-height:260px;overflow-y:auto;
          color:#94a3b8;white-space:pre-wrap;word-break:break-all;
          font-family:'Courier New',monospace;
        ">Connecting...</pre>
      </div>
    `;
    document.body.appendChild(el);

    if (!document.getElementById('diag-styles')) {
      const style = document.createElement('style');
      style.id = 'diag-styles';
      style.textContent = `
        .diag-card {
          background:#0f172a;border:1px solid #334155;border-radius:.6rem;padding:.85rem;
        }
        .diag-label {
          font-size:.75rem;color:#94a3b8;font-weight:500;text-transform:uppercase;
          letter-spacing:.04em;margin-bottom:.25rem;
        }
        .diag-hz {
          font-size:1.5rem;font-weight:700;color:#22c55e;
        }
        .diag-hz.warn { color:#eab308; }
        .diag-hz.bad  { color:#ef4444; }
        code { background:#334155;padding:.1rem .3rem;border-radius:.25rem; }
      `;
      document.head.appendChild(style);
    }

    el.addEventListener('click', function (e) {
      if (e.target === el) DiagnosticsModal.close();
    });
  }

  function _syncEndpointSelect() {
    const select = document.getElementById('diag-endpoint-select');
    if (select) select.value = _endpoint;
  }

  function _connectWs() {
    if (_ws && (_ws.readyState === WebSocket.OPEN || _ws.readyState === WebSocket.CONNECTING)) return;
    try {
      _shouldReconnect = true;
      const token = ++_wsToken;
      _ws = new WebSocket(_diagWsUrl());
      _ws.onmessage = function (ev) {
        if (token !== _wsToken) return;
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'stats') _applyStats(msg);
          else if (msg.type === 'packet') _appendPacket(msg);
        } catch (_) {}
      };
      _ws.onerror = function () {
        if (token !== _wsToken) return;
        _appendRawLine('[WS error - will retry]');
      };
      _ws.onclose = function () {
        if (token !== _wsToken) return;
        _ws = null;
        if (_shouldReconnect) {
          _appendRawLine('[WS closed - reconnecting in 3s]');
          setTimeout(_connectWs, 3000);
        }
      };
      _ws.onopen = function () {
        if (token !== _wsToken) return;
        _appendRawLine('[WS connected]');
      };
    } catch (e) {
      _appendRawLine('[Could not open WS: ' + e.message + ']');
    }
  }

  function _disconnectWs() {
    _shouldReconnect = false;
    _wsToken += 1;
    if (_ws) {
      try { _ws.close(); } catch (_) {}
      _ws = null;
    }
  }

  function _fetchInitialStats() {
    fetch(_statsUrl(), { cache: 'no-store' })
      .then(r => r.json())
      .then(d => { d.type = 'stats'; _applyStats(d); })
      .catch(() => {});
  }

  function _applyStats(stats) {
    const wsUrlEl = document.getElementById('diag-ws-url');
    if (wsUrlEl) wsUrlEl.textContent = stats.ws_url_hint || `ws://<LAN-IP>:8000${_endpointPath()}`;

    const lanEl = document.getElementById('diag-lan-ips');
    if (lanEl) {
      const ips = stats.server_lan_ips || [];
      lanEl.textContent = ips.length ? ips.join(', ') : '(none found)';
      lanEl.dataset.ips = JSON.stringify(ips);
    }

    const activeEl = document.getElementById('diag-active-session');
    if (activeEl) {
      activeEl.textContent = stats.active_session
        ? `Recording: ${stats.active_session}`
        : 'No active session';
      activeEl.style.color = stats.active_session ? '#ef4444' : '#94a3b8';
    }

    const timeEl = document.getElementById('diag-server-time');
    if (timeEl && stats.server_time_ms) {
      timeEl.textContent = new Date(stats.server_time_ms).toLocaleTimeString();
    }

    _renderGloveStats('right', 1, stats);
    _renderGloveStats('left', 2, stats);
  }

  function _renderGloveStats(side, hid, stats) {
    const hz = (stats.observed_hz || {})[hid];
    const age = (stats.last_seen_age_ms || {})[hid];
    const schema = (stats.schema || {})[String(hid)] || {};

    const hzEl = document.getElementById(`diag-hz-${side}`);
    if (hzEl) {
      const hzVal = hz !== undefined ? hz : '-';
      hzEl.textContent = `${hzVal} Hz`;
      hzEl.className = 'diag-hz';
      if (hz !== undefined) {
        if (hz < 25) hzEl.classList.add('bad');
        else if (hz < 45) hzEl.classList.add('warn');
      }
    }

    const ageEl = document.getElementById(`diag-age-${side}`);
    if (ageEl) {
      ageEl.textContent = age !== null && age !== undefined
        ? `last seen ${age}ms ago`
        : 'never seen';
    }

    const schemaEl = document.getElementById(`diag-schema-${side}`);
    if (schemaEl) {
      const err = schema.last_error ? ` | error: ${schema.last_error}` : '';
      schemaEl.textContent = `ok ${schema.valid || 0}  bad ${schema.invalid || 0}${err}`;
      schemaEl.style.color = (schema.invalid > 0) ? '#ef4444' : '#22c55e';
    }
  }

  function _appendPacket(msg) {
    if (_paused) return;
    const hid = (msg.packet || {}).hand_id;
    if (_handFilter !== 0 && hid !== _handFilter) return;

    const side = hid === 1 ? 'R' : hid === 2 ? 'L' : '?';
    const valid = msg.valid ? 'ok' : `bad ${msg.error || ''}`;
    const p = msg.packet || {};
    const ts = p.timestamp_ms || '?';
    const flex = (p.flex || []).map(v => Math.round(v)).join(',');
    const fsr  = (p.fsr  || []).map(v => Math.round(v)).join(',');

    // IMU fields present in every prediction-mode packet
    const quat  = p.quaternion ? p.quaternion.map(v => v.toFixed(2)).join(',') : null;
    const accel = p.accel      ? p.accel.map(v => v.toFixed(1)).join(',')      : null;
    const gyro  = p.gyro       ? p.gyro.map(v => v.toFixed(1)).join(',')       : null;
    const imuStr = quat ? ` | q=[${quat}] a=[${accel}] g=[${gyro}]` : '';

    _appendRawLine(`[${side}] ts=${ts} flex=[${flex}] fsr=[${fsr}]${imuStr} ${valid}`);
  }

  function _appendRawLine(line) {
    _packetLog.push(line);
    if (_packetLog.length > MAX_LOG) _packetLog.shift();
    const el = document.getElementById('diag-packet-log');
    if (el) {
      el.textContent = _packetLog.join('\n');
      el.scrollTop = el.scrollHeight;
    }
  }

  function _resetLog() {
    _packetLog = [];
    const logEl = document.getElementById('diag-packet-log');
    if (logEl) logEl.textContent = 'Connecting...';
  }

  const DiagnosticsModal = {
    init() {
      _injectModal();
      _syncEndpointSelect();
    },

    open() {
      _injectModal();
      _syncEndpointSelect();
      const modal = document.getElementById('diag-modal');
      if (modal) modal.style.display = 'flex';
      _resetLog();
      _connectWs();
      _fetchInitialStats();
    },

    close() {
      const modal = document.getElementById('diag-modal');
      if (modal) modal.style.display = 'none';
      _disconnectWs();
    },

    _togglePause() {
      _paused = !_paused;
      const btn = document.getElementById('diag-pause-btn');
      if (btn) btn.textContent = _paused ? 'Resume' : 'Pause';
    },

    _setFilter(val) {
      _handFilter = parseInt(val, 10) || 0;
    },

    _setEndpoint(val) {
      _endpoint = val === 'predict' ? 'predict' : 'collect';
      _resetLog();
      _disconnectWs();
      _connectWs();
      _fetchInitialStats();
    },

    _copyLog() {
      const text = _packetLog.join('\n');
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).catch(() => {});
      }
    },

    _copyLanIp() {
      const el = document.getElementById('diag-lan-ips');
      if (!el) return;
      const ips = JSON.parse(el.dataset.ips || '[]');
      if (ips.length && navigator.clipboard) {
        navigator.clipboard.writeText(`ws://${ips[0]}:8000${_endpointPath()}`).catch(() => {});
      }
    },
  };

  global.DiagnosticsModal = DiagnosticsModal;
})(window);
