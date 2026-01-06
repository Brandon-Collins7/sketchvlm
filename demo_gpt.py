# ------------- HTML -------------
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Sketch Grid Demo</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 20px; color: #111; background: #f9fafb; }
    .container { display: grid; grid-template-columns: 380px 1fr; gap: 20px; }
    .left, .right { background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    label { font-weight: 600; display: block; margin-top: 8px; }
    .row { margin-bottom: 12px; }
    .legend { font-size: 12px; color: #555; margin-top: 4px; }
    .btn { appearance: none; background: #111; color: #fff; border: none; border-radius: 10px; padding: 10px 14px; cursor: pointer; font-size: 14px; transition: opacity 0.2s; }
    .btn.alt { background: #2563eb; }
    .btn.success { background: #16a34a; }
    .btn.ghost { background: #f3f4f6; color: #111; border: 1px solid #e5e7eb; }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn:hover:not(:disabled) { opacity: 0.9; }
    .btn + .btn { margin-left: 8px; }
    .stage { position: relative; display: inline-block; background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 10px; }
    .stage img { display: block; max-width: 100%; height: auto; border-radius: 8px; }
    .svg-overlay { position: absolute; left: 10px; top: 10px; pointer-events: none; }
    .svg-overlay.interactive { pointer-events: auto; }
    .preview-layer { pointer-events: none; }
    .preview-layer .pending-preview { cursor: move; }

    .preview-layer .pending-preview text { 
      opacity: 0.8;
      font-weight: bold;
    }
    .row.flex { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    select, textarea, input[type="text"] { width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; padding: 8px; font-size: 14px; }
    .status { padding: 8px; background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; font-size: 13px; }
    .drawing-controls { display: none; background: #fef3c7; border: 1px solid #fde047; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
    .drawing-controls.active { display: block; }
    .samples-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; margin-top: 8px; }
    .sample-btn { padding: 8px; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; cursor: pointer; font-size: 12px; text-align: center; transition: all 0.2s; }
    .sample-btn:hover { background: #e5e7eb; transform: translateY(-1px); }
    .sample-btn img { width: 100%; height: 60px; object-fit: cover; border-radius: 4px; margin-bottom: 4px; }
    .debug-panel { margin-top: 16px; padding: 12px; background: #fef3c7; border: 1px solid #fde047; border-radius: 8px; max-height: 400px; overflow-y: auto; }
    .debug-panel h4 { margin: 0 0 8px 0; font-size: 14px; }
    .debug-panel pre { font-size: 11px; background: #fffbeb; padding: 8px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
    .debug-toggle { cursor: pointer; user-select: none; }
    .grid-info { font-size: 12px; color: #666; background: #f9fafb; padding: 8px; border-radius: 6px; margin-top: 8px; }
    
    
    :root{
      --bg: #0b0f17;
      --card: #0f1729;
      --muted: #1b253b;
      --text: #e6edf6;
      --text-dim: #9fb2c8;
      --primary: #3b82f6;
      --success: #16a34a;
      --warn: #f59e0b;
      --ring: #60a5fa;
    }

    /* Card polish */
    body{ background: var(--bg); color: var(--text); }
    .left,.right{
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
      border: 1px solid rgba(255,255,255,0.06);
      box-shadow: 0 10px 30px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.03);
      backdrop-filter: blur(3px);
    }
    label{ color: var(--text); }
    .legend{ color: var(--text-dim); }

    /* Buttons */
    .btn{
      background: linear-gradient(180deg, #1f2937, #111827);
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 1px 0 rgba(255,255,255,0.05), 0 6px 20px rgba(0,0,0,0.25);
      transition: transform .08s ease, box-shadow .2s ease, background .2s ease, filter .2s ease;
    }
    .btn:hover:not(:disabled){ transform: translateY(-1px); box-shadow: 0 8px 24px rgba(0,0,0,0.35); }
    .btn:active:not(:disabled){ transform: translateY(0); }

    .btn.alt{
      background: linear-gradient(180deg, #2563eb, #1e40af);
      border-color: rgba(37,99,235,0.6);
    }
    .btn.success{
      background: linear-gradient(180deg, #16a34a, #166534);
      border-color: rgba(22,163,74,0.6);
    }
    .btn.ghost{
      background: linear-gradient(180deg, #111827, #0b1220);
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.08);
    }

    /* ✨ Toggle feel for Move button */
    #btnMove{
      position: relative;
      isolation: isolate;
    }
    #btnMove.active{
      background: linear-gradient(180deg, #0ea5e9, #0369a1);
      border-color: rgba(14,165,233,0.7);
      box-shadow: 0 0 0 3px rgba(14,165,233,0.25), 0 10px 30px rgba(14,165,233,0.35);
      filter: brightness(1.02);
    }
    #btnMove:not(.active){
      filter: brightness(.85);
    }
    #btnMove::after{
      /* small LED pill */
      content: attr(data-state);
      position: absolute;
      right: -10px;
      top: -10px;
      padding: 2px 8px;
      font-size: 10px;
      border-radius: 999px;
      background: rgba(148,163,184,0.2);
      color: #cbd5e1;
      border: 1px solid rgba(255,255,255,0.08);
    }
    #btnMove.active::after{
      background: rgba(34,197,94,0.2);
      color: #bbf7d0;
      border-color: rgba(34,197,94,0.3);
    }

    /* Inputs */
    select, textarea, input[type="text"], input[type="number"], input[type="color"]{
      background: #0b1220;
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    select:focus, textarea:focus, input:focus{
      outline: none;
      box-shadow: 0 0 0 3px rgba(96,165,250,0.25);
      border-color: rgba(96,165,250,0.6);
    }

    /* Stage polish */
    .stage{
      background: #0b1220;
      border: 1px solid rgba(255,255,255,0.06);
      box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .stage img{ border-radius: 10px; }

    /* Status */
    .status{
      background: rgba(59,130,246,0.08);
      border: 1px solid rgba(59,130,246,0.25);
      color: var(--text);
    }

    /* Pending preview stays slightly translucent but never stacks opacity */
    .preview-layer .pending-preview{
      opacity: .82;
    }
    .preview-layer .pending-preview *{
      opacity: 1 !important;  /* prevent cumulative fading on clones */
    }

    /* Tiny toolbar spacing and grouping */
    .row.flex .btn{ border-radius: 12px; }

  </style>
</head>
<body>
  <h2>Sketch Grid Demo - Multi/One-turn Annotation</h2>
  <div class="container">
    <div class="left">
      <div class="row">
        <label>Source Image</label>
        <form id="uploadForm">
          <input type="file" name="image" accept="image/*"/>
          <button class="btn" type="submit">Upload</button>
        </form>
        <div id="gridInfo" class="grid-info" style="display:none;"></div>
        
        <div style="margin-top: 12px;">
          <label style="font-size: 13px;">Sample Images</label>
          <div id="samplesGrid" class="samples-grid"></div>
        </div>
      </div>
      
      <div class="row">
        <label>Task</label>
        <select id="task">
          <option value="counting">Counting</option>
          <option value="labeling">Labeling</option>
          <option value="custom">Custom Prompt</option>
        </select>
        <label>Custom prompt</label>
        <textarea id="customPrompt" rows="3" placeholder="For counting: object name (e.g., 'cars')
For labeling: concept name (e.g., 'bicycle')
For custom: your instruction"></textarea>
        <div class="legend">Counting: count objects. Labeling: label parts. Custom: any instruction.</div>
      </div>
      
      <div class="row">
        <label>Mode</label>
        <select id="mode">
          <option value="multi_turn">Multi-turn (step-by-step approval)</option>
          <option value="one_turn">One-turn (all at once)</option>
        </select>
        <div class="legend">Multi-turn: Review each stroke. One-turn: Place all immediately.</div>
      </div>
      
      <div class="row flex">
        <label><input type="checkbox" id="showGrid" checked/> Show grid overlay</label>
      </div>
      
      <div class="row">
        <button class="btn alt" id="btnCall">Call Model</button>
        <button class="btn ghost" id="btnDone">Reset</button>
      </div>
      
      <div id="multiTurnControls" style="display: none;">
        <div class="row">
          <div class="status" id="status"></div>
        </div>
        
        <div class="drawing-controls" id="drawingControls">
          <strong style="font-size: 13px;">Add Your Own Stroke:</strong>
          <div class="row flex" style="margin-top: 8px;">
            <button class="btn ghost" id="btnDrawLine">✏️ Draw Line</button>
            <button class="btn ghost" id="btnAddText">📝 Add Text</button>
            <button class="btn ghost" id="btnFreeDraw" title="Freehand drawing">🖊️ Free Draw</button>
            <button class="btn ghost" id="btnCancelDraw">✖ Cancel</button>
          </div>
          <div id="textInput" style="display:none; margin-top: 8px;">
            <input type="text" id="textValue" placeholder="Enter text..." style="margin-bottom: 4px;"/>
            <button class="btn" id="btnPlaceText">Place Text</button>
          </div>
        </div>
        
        <div class="row flex" style="margin-top: 8px; gap:12px;">
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:12px;">Line</span>
            <input type="color" id="lineColor" value="#00ff00" title="Line color" />
            <input type="number" id="lineWidth" value="2" min="1" max="12" style="width:64px" title="Line width"/>
          </div>
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:12px;">Text</span>
            <input type="color" id="textColor" value="#0066ff" title="Text color" />
            <input type="number" id="textSize" value="16" min="8" max="72" style="width:64px" title="Text size (px)"/>
          </div>
        </div>
        
        <div class="row">
          <button class="btn success" id="btnAccept">✓ Accept Stroke</button>
        </div>
        <div class="row">
          <button class="btn ghost" id="btnMove" title="Toggle move mode">🖐 Move</button>
          <button class="btn ghost" id="btnAddOwn">➕ Add My Own</button>
          <button class="btn" id="btnAcceptAll">Accept All Remaining</button>
          <button class="btn alt" id="btnRequestNew">Request New Call</button>
        </div>
        <div class="legend">Move: drag pending (orange) or any of your strokes. Add My Own: draw lines, text, or freehand.</div>
      </div>
      
      <div class="row">
        <button class="btn ghost" id="btnUndo" disabled>↶ Undo Last</button>
        <button class="btn ghost" id="btnDownload">💾 Download</button>
      </div>
      
      <div class="row">
        <div class="debug-toggle" id="debugToggle">
          <strong>🔍 Debug Info</strong> (click to toggle)
        </div>
        <div id="debugPanel" class="debug-panel" style="display: none;">
          <div id="debugContent">No debug info yet. Call the model first.</div>
        </div>
      </div>
      
      <div style="margin-top:8px;">
        <label style="font-size:13px; display:flex; align-items:center; gap:6px;">
          <input type="checkbox" id="toggleAcceptedXml" />
          Show accepted XML sent on next turn
        </label>
        <pre id="acceptedXmlBox" style="display:none; margin-top:6px; max-height:220px; overflow:auto; background:#0f0f0f; color:#ddd; padding:8px; border-radius:6px;"></pre>
      </div>

      
      <div class="row" id="answerRow" style="display:none;">
        <label>Model Answer</label>
        <div id="finalAnswer" class="status"></div>
      </div>

    </div>
    
    <div class="right">
      <div class="stage">
        <img id="srcImg" src="" alt="Upload an image to start"/>
        <div id="svgContainer" class="svg-overlay"></div>
      </div>
    </div>
  </div>

  <script>
    const $ = (q) => document.querySelector(q);
    
    const state = {
      dataURL: '',
      mode: 'multi_turn',
      showGrid: true,
      fullSVG: '',
      pendingBlock: null,
      pendingPreview: null,  // SVG for pending stroke preview
      queueRemaining: 0,
      canUndo: false,
      isDraggingPending: false,
      dragOffset: {x: 0, y: 0},
      isDrawingMode: false,
      drawingType: null,  // 'line' or 'text'
      drawStart: null,
      currentLine: null,
      gridPx: { w: null, h: null },
    };
    
    
    // ---- Modes: move | freehand | add_line | add_text ----
    const Modes = {
      MOVE: 'move',
      FREEHAND: 'freehand',
      ADD_LINE: 'add_line',
      ADD_TEXT: 'add_text',
    };


    function syncAcceptedXmlUI() {
      const box = document.getElementById('acceptedXmlBox');
      const toggle = document.getElementById('toggleAcceptedXml');
      if (!box || !toggle) return;
      const show = toggle.checked;
      box.style.display = show ? 'block' : 'none';
      box.textContent = show ? (state.acceptedXmlText || '(none yet)') : '';
    }
    document.getElementById('toggleAcceptedXml')?.addEventListener('change', syncAcceptedXmlUI);


    // modes: 'move' | 'freehand' | 'add_line' | 'add_text' | 'multi_turn'
    state.mode = 'multi_turn';   // keep this for multi-turn logic
    state.moveMode = false;      // new: separate "move" toggle

    function setMove(on) {
      state.moveMode = !!on;
      const btn = $('#btnMove');
      if (btn) {
        btn.classList.toggle('active', state.moveMode);
        btn.setAttribute('aria-pressed', String(state.moveMode));
        btn.setAttribute('data-state', state.moveMode ? 'ON' : 'OFF');
        // Optional text tweak (keeps emoji the same)
        // btn.textContent = state.moveMode ? '🖐 Move (On)' : '🖐 Move';
      }
      // When move is ON, kill any drawing tool
      if (state.moveMode) {
        state.isDrawingMode = false;
        state.drawingType = null;
        state.drawStart = null;
        $('#textInput')?.style?.setProperty('display','none');
        $('#drawingControls')?.classList?.remove('active');
      }
      updateDisplay();
    }

    const isMoveMode = () => state.moveMode === true;

    // clicking Move toggles it
    $('#btnMove')?.addEventListener('click', () => setMove(!state.moveMode));



    
    function updateDisplay() {
      const img = $('#srcImg');
      const container = $('#svgContainer');

      // Clear overlay container each render
      container.innerHTML = '';

      if (img.complete && img.naturalWidth) {
        const imgWidth = img.clientWidth;
        const imgHeight = img.clientHeight;

        // ----- MAIN SVG (accepted strokes) -----
        const mainSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        mainSvg.setAttribute('width', imgWidth);
        mainSvg.setAttribute('height', imgHeight);
        mainSvg.style.width = imgWidth + 'px';
        mainSvg.style.height = imgHeight + 'px';

        // Default viewBox (will be overridden by source if present)
        let viewBoxWidth = imgWidth;
        let viewBoxHeight = imgHeight;

        if (state.fullSVG) {
          const tempDiv = document.createElement('div');
          tempDiv.innerHTML = state.fullSVG;
          const sourceSvg = tempDiv.querySelector('svg');
          if (sourceSvg) {
            viewBoxWidth  = parseFloat(sourceSvg.getAttribute('width'))  || imgWidth;
            viewBoxHeight = parseFloat(sourceSvg.getAttribute('height')) || imgHeight;
            mainSvg.setAttribute('viewBox', `0 0 ${viewBoxWidth} ${viewBoxHeight}`);

            // Copy all accepted children
            Array.from(sourceSvg.children).forEach(child => {
              mainSvg.appendChild(child.cloneNode(true));
            });
          }
        } else {
          // Lock to server grid pixels if known
          const gx = state.gridPx?.w || imgWidth;
          const gy = state.gridPx?.h || imgHeight;
          viewBoxWidth = gx;
          viewBoxHeight = gy;
          mainSvg.setAttribute('viewBox', `0 0 ${gx} ${gy}`);
        }

        container.appendChild(mainSvg);

        // Add pending preview as a separate overlay SVG (same viewBox as main)
        if (state.pendingPreview && state.pendingBlock) {
          const previewSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          previewSvg.setAttribute('width', imgWidth);
          previewSvg.setAttribute('height', imgHeight);
          previewSvg.style.width = imgWidth + 'px';
          previewSvg.style.height = imgHeight + 'px';
          previewSvg.style.position = 'absolute';
          previewSvg.style.top = '0';
          previewSvg.style.left = '0';
          previewSvg.classList.add('preview-layer');

          // CRITICAL: overlay must not block clicks except on the pending group itself
          previewSvg.style.pointerEvents = 'none';

          // Match the main SVG’s viewBox 1:1 to avoid any offset
          const mainSvgForVB = container.querySelector('svg:not(.preview-layer)');
          if (mainSvgForVB) {
            const vb = mainSvgForVB.getAttribute('viewBox');
            if (vb) previewSvg.setAttribute('viewBox', vb);
          }

          // Parse the pending preview SVG we got from the server
          const tempDiv = document.createElement('div');
          tempDiv.innerHTML = state.pendingPreview;
          const sourceSvg = tempDiv.querySelector('svg');

          if (sourceSvg) {
            const previewGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            previewGroup.classList.add('pending-preview');
            previewGroup.setAttribute('data-pending', 'true');

            // Only the pending group should accept pointer events when in move mode.
            // The rest of the overlay stays transparent to pointer hit-testing.
            previewGroup.style.pointerEvents = isMoveMode() ? 'auto' : 'none';

            // Orange stroke/fill at full opacity (no cumulative fading)
            Array.from(sourceSvg.children).forEach(child => {
              const cloned = child.cloneNode(true);
              if (cloned.hasAttribute('stroke')) {
                cloned.setAttribute('stroke', '#ff6b00');
                cloned.setAttribute('stroke-opacity', '1.0');
              }
              if (cloned.hasAttribute('fill') && cloned.getAttribute('fill') !== 'none') {
                cloned.setAttribute('fill', '#ff6b00');
                cloned.setAttribute('fill-opacity', '1.0');
              }
              previewGroup.appendChild(cloned);
            });

            previewSvg.appendChild(previewGroup);
            container.appendChild(previewSvg);
          }
        }


        // Interactivity class only once
        container.classList.toggle('interactive', isMoveMode() || state.isDrawingMode);
      }

      // ----- Status / buttons (multi-turn panel) -----
      $('#multiTurnControls').style.display = 'block';

      const hasPending = !!(state.pendingBlock && state.pendingPreview);
      const q = state.queueRemaining || 0;

      $('#status').textContent = hasPending
        ? `Pending stroke. ${q} remaining in queue.`
        : (q > 0 ? `Queue has ${q} more strokes.` : 'No pending strokes. You can still add your own strokes.');

      // Accept button only depends on pending availability
      $('#btnAccept').disabled = !hasPending;
      const btnAcceptAll = $('#btnAcceptAll');
      if (btnAcceptAll) btnAcceptAll.disabled = !hasPending && q === 0;
     

      // Undo state
      $('#btnUndo').disabled = !state.canUndo;
    }

    
    async function api(path, payload) {
      const r = await fetch(path, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload || {})
      });
      return await r.json();
    }
    
    async function loadDebugInfo() {
      const res = await api('/get_debug_info', {});
      if (res.ok && res.debug) {
        const d = res.debug;
        let html = '<h4>Last API Call</h4>';
        html += `<p><strong>Timestamp:</strong> ${d.timestamp}</p>`;
        html += `<p><strong>Grid:</strong> ${d.grid_info?.res_x || 'N/A'}x${d.grid_info?.res_y || 'N/A'} cells @ ${d.grid_info?.cell_size || 'N/A'}px</p>`;
        html += `<h4>System Prompt</h4><pre>${escapeHtml(d.system_prompt || 'N/A').substring(0, 800)}...</pre>`;
        html += `<h4>User Prompt</h4><pre>${escapeHtml(d.user_prompt || 'N/A')}</pre>`;
        if (d.error) {
          html += `<h4 style="color:red;">Error</h4><pre>${escapeHtml(d.error)}</pre>`;
        }
        if (d.model_output) {
          html += `<h4>Model Output</h4><pre>${escapeHtml(d.model_output).substring(0, 2000)}...</pre>`;
          const fa = (d.model_output.match(/<final_answer>\s*([\s\S]*?)\s*<\/final_answer>/i) || [])[1];
          if (fa) {
            $('#finalAnswer').textContent = fa.trim();
            $('#answerRow').style.display = 'block';
          }
        }

        $('#debugContent').innerHTML = html;
      } else {
        $('#debugContent').innerHTML = 'No debug info available.';
      }
    }
    
    function escapeHtml(text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }
    
    async function loadSamples() {
      try {
        const res = await fetch('/samples');
        const data = await res.json();
        const grid = $('#samplesGrid');
        grid.innerHTML = '';

        if (!data.ok) return;

        // Prefer grouped response
        if (data.groups && data.groups.length) {
          for (const group of data.groups) {
            // Group header
            const header = document.createElement('div');
            header.style.fontWeight = '700';
            header.style.margin = '8px 0 4px 0';
            header.style.fontSize = '13px';
            header.textContent = group.name;
            grid.appendChild(header);

            // Group items
            const groupWrap = document.createElement('div');
            groupWrap.className = 'samples-grid';
            grid.appendChild(groupWrap);

            for (const sample of group.items) {
              const btn = document.createElement('div');
              btn.className = 'sample-btn';
              btn.innerHTML = `<img src="${sample.data_url}" alt="${sample.name}"/><div>${sample.name}</div>`;
              btn.onclick = () => loadSample(sample);  // sample has .task and .prompt
              groupWrap.appendChild(btn);
            }
          }
          return;
        }

        // Fallback: legacy flat list at data.samples
        if (data.samples && data.samples.length) {
          for (const sample of data.samples) {
            const btn = document.createElement('div');
            btn.className = 'sample-btn';
            btn.innerHTML = `<img src="${sample.data_url}" alt="${sample.name}"/><div>${sample.name}</div>`;
            btn.onclick = () => loadSample(sample);
            grid.appendChild(btn);
          }
        }
      } catch (e) {
        console.log('No samples available:', e);
      }
    }

    
    async function loadSample(sample) {
      // 1) Switch Task dropdown based on sample.task (counting | labeling | custom)
      if (sample.task) {
        $('#task').value = sample.task;
        await api('/set-task', {
          task: sample.task,
          prompt: sample.prompt || ''
        });
      }

      // 2) Autofill custom prompt box (still helpful for counting/labeling samples too)
      $('#customPrompt').value = sample.prompt || '';

      // 3) Reset local UI state
      state.dataURL = sample.data_url;
      state.fullSVG = '';
      state.pendingBlock = null;
      state.pendingPreview = null;
      state.queueRemaining = 0;
      state.canUndo = false;
      $('#srcImg').src = state.dataURL;

      // 4) Upload to backend so it returns the grid/no-grid canvas (respects checkbox)
      const uploadRes = await api('/upload', {
        image_data: sample.data_url,
        show_grid: state.showGrid
      });

      if (uploadRes.ok) {
        if (uploadRes.data_url) {
          state.dataURL = uploadRes.data_url;
          $('#srcImg').src = state.dataURL;
        }
        if (uploadRes.grid_info) {
          const gi = uploadRes.grid_info;
          state.gridPx = gi.grid_px || state.gridPx;
          $('#gridInfo').innerHTML = `Grid: ${gi.res_x}×${gi.res_y} cells (${gi.cell_size}px each)`;
          $('#gridInfo').style.display = 'block';
        }
        $('#srcImg').onload = () => updateDisplay();
      }
    }
    
    
    function setupDrawingHandlers() {
      const container = $('#svgContainer');
      
      container.addEventListener('click', (e) => {
        if (!state.isDrawingMode) return;
        
        // Get the main SVG (not the preview layer)
        let svg = null;
        for (const child of container.children) {
          if (child.tagName === 'svg' && !child.classList.contains('preview-layer')) {
            svg = child;
            break;
          }
        }
        
        if (!svg) {
          console.error('No main SVG found');
          return;
        }
        
        const rect = container.getBoundingClientRect();
        const viewBox = svg.getAttribute('viewBox');
        let viewBoxValues = viewBox ? viewBox.split(' ').map(parseFloat) : [0, 0, svg.clientWidth, svg.clientHeight];
        
        const scaleX = viewBoxValues[2] / svg.clientWidth;
        const scaleY = viewBoxValues[3] / svg.clientHeight;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        
        if (state.drawingType === 'line') {
          if (!state.drawStart) {
            state.drawStart = {x, y};
            // Show temporary dot
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', x);
            circle.setAttribute('cy', y);
            circle.setAttribute('r', '5');
            circle.setAttribute('fill', 'red');
            circle.setAttribute('data-temp', 'true');
            svg.appendChild(circle);
          } else {
            // Complete line
            completeLine(state.drawStart.x, state.drawStart.y, x, y);
            state.drawStart = null;
            state.isDrawingMode = false;
            $('#drawingControls').classList.remove('active');
          }
        } else if (state.drawingType === 'text') {
          state.drawStart = {x, y};
          $('#textInput').style.display = 'block';
          $('#textValue').focus();
        }
        
        e.preventDefault();
        e.stopPropagation();
      });
    }
    
    (function setupMoveHandlers(){
      const container = $('#svgContainer');

      // --- drag PENDING (model) stroke when in move mode ---
      let pDrag = { active:false, start:{x:0,y:0}, base:{x:0,y:0} };

      container.addEventListener('pointerdown', (e) => {
        if (!isMoveMode()) return;

        const previewLayer = container.querySelector('.preview-layer');
        const pending = previewLayer?.querySelector('[data-pending="true"]');
        if (!pending) return;

        const hit = e.target.closest('[data-pending="true"]');
        if (!hit) return; // must click the pending stroke

        const mainSvg = container.querySelector('svg:not(.preview-layer)');
        if (!mainSvg) return;

        // base = current translate on <g>
        const t = pending.getAttribute('transform') || '';
        const m = t.match(/translate\(([-\d.]+),\s*([-\d.]+)\)/);
        pDrag.base = m ? {x:parseFloat(m[1]), y:parseFloat(m[2])} : {x:0,y:0};

        // cursor in SVG coords
        pDrag.start = svgPointFromClient(e, mainSvg);
        pending.setPointerCapture?.(e.pointerId);
        pDrag.active = true;
        e.preventDefault();
      });

      container.addEventListener('pointermove', (e) => {
        if (!pDrag.active) return;
        const mainSvg = container.querySelector('svg:not(.preview-layer)');
        if (!mainSvg) return;
        const curr = svgPointFromClient(e, mainSvg);
        const dx = curr.x - pDrag.start.x;
        const dy = curr.y - pDrag.start.y;
        const pending = container.querySelector('.preview-layer [data-pending="true"]');
        if (pending) pending.setAttribute('transform', `translate(${pDrag.base.x + dx}, ${pDrag.base.y + dy})`);
      });

      ['pointerup','pointerleave','pointercancel'].forEach(ev => {
        container.addEventListener(ev, async () => {
          if (!pDrag.active) return;
          pDrag.active = false;

          const pending = container.querySelector('.preview-layer [data-pending="true"]');
          if (!pending) return;
          const t = pending.getAttribute('transform') || '';
          const m = t.match(/translate\(([-\d.]+),\s*([-\d.]+)\)/);

           // Send ONLY the delta of this drag:
            const currX = m ? parseFloat(m[1]) : 0;
            const currY = m ? parseFloat(m[2]) : 0;
            const dx = currX - uDrag.base.x;
            const dy = currY - uDrag.base.y;

          try {
            const res = await fetch('/update_pending_position', {
              method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({ delta_x: dx, delta_y: dy })
            }).then(r=>r.json());
            if (res.ok) {
              state.pendingPreview = res.updated_svg || state.pendingPreview;
              updateDisplay();
            }
          } catch(_) {}
        });
      });


      async function commitPendingDelta() {
        const previewLayer = container.querySelector('.preview-layer');
        const pending = previewLayer?.querySelector('[data-pending="true"]');
        if (!pending) return;
        const t = pending.getAttribute('transform') || '';
        const m = t.match(/translate\(([-\d.]+),\s*([-\d.]+)\)/);
        const dx = m ? parseFloat(m[1]) : 0, dy = m ? parseFloat(m[2]) : 0;
        const res = await fetch('/update_pending_position', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ delta_x: dx, delta_y: dy })
        }).then(r=>r.json());
        if (res.ok) {
          // server bakes transform into new base; refresh the preview svg
          state.pendingPreview = res.updated_svg || state.pendingPreview;
          updateDisplay();
        }
      }


      // --- drag USER strokes (g.user-stroke) in move mode ---
      let uDrag = { el:null, start:{x:0,y:0}, base:{x:0,y:0} };

      container.addEventListener('pointerdown', (e) => {
        if (!isMoveMode()) return;
        // ignore the pending overlay entirely
        if (e.target.closest('.preview-layer')) return;

        const g = e.target.closest('g.user-stroke');
        if (!g) return;

        const mainSvg = container.querySelector('svg:not(.preview-layer)');
        if (!mainSvg) return;

        uDrag.el = g;

        // read current translate on this <g>
        const t = g.getAttribute('transform') || '';
        const m = t.match(/translate\(([-\d.]+),\s*([-\d.]+)\)/);
        uDrag.base = m ? {x:parseFloat(m[1]), y:parseFloat(m[2])} : {x:0,y:0};

        // start cursor in SVG coords
        uDrag.start = svgPointFromClient(e, mainSvg);

        g.setPointerCapture?.(e.pointerId);
        e.preventDefault();
        e.stopPropagation();
      });

      container.addEventListener('pointermove', (e) => {
        if (!uDrag.el) return;
        const mainSvg = container.querySelector('svg:not(.preview-layer)');
        if (!mainSvg) return;

        const curr = svgPointFromClient(e, mainSvg);
        const dx = curr.x - uDrag.start.x;
        const dy = curr.y - uDrag.start.y;

        uDrag.el.setAttribute('transform', `translate(${uDrag.base.x + dx}, ${uDrag.base.y + dy})`);
      });

      ['pointerup','pointerleave','pointercancel'].forEach(ev => {
        container.addEventListener(ev, async () => {
          if (!uDrag.el) return;

          const g = uDrag.el;
          uDrag.el = null;

          const t = g.getAttribute('transform') || '';
          const m = t.match(/translate\(([-\d.]+),\s*([-\d.]+)\)/);
          const dx = m ? parseFloat(m[1]) : 0;
          const dy = m ? parseFloat(m[2]) : 0;
          const id = g.getAttribute('id');

          try {
            const res = await fetch('/transform_user_stroke', {
              method:'POST', headers:{'Content-Type':'application/json'},
              body: JSON.stringify({ id, dx, dy })
            }).then(r=>r.json());
            if (res.ok && res.full_svg) {
              state.fullSVG = res.full_svg; // server baked the transform
              updateDisplay();
            }
          } catch {}
        });
      });


      // Delete only in move mode
      window.addEventListener('keydown', async (e) => {
        if (!isMoveMode()) return;
        if (!['Delete','Backspace'].includes(e.key)) return;
        // prefer a selected element; fallback to element under cursor is out-of-scope here
        const sel = container.querySelector('g.user-stroke.selected');
        if (!sel) return;
        const id = sel.getAttribute('id');
        const res = await fetch('/delete_user_stroke', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ stroke_id: id })
        }).then(r=>r.json());
        if (res.ok) {
          state.fullSVG = res.full_svg;
          state.canUndo = !!res.can_undo;
          updateDisplay();
        }
      });
    })();

    



    
    async function completeLine(x1, y1, x2, y2) {
      const res = await api('/add_user_stroke', {
        type: 'line',
        points: [[x1, y1], [x2, y2]],
        line_color: $('#lineColor')?.value || '#00ff00',
        line_width: parseFloat($('#lineWidth')?.value || '2')
      });
      console.log('add_user_stroke(line):', res);
      if (res.ok) {
        // remove any temp markers from main svg
        const svg = $('#svgContainer svg');
        if (svg) svg.querySelectorAll('[data-temp]').forEach(el => el.remove());
        // existing:
        state.fullSVG = res.full_svg || state.fullSVG;
        state.isDrawingMode = false;
        state.drawingType = null;
        state.drawStart = null;
        updateDisplay();
      }
    }
    
    async function completeText(x, y, text) {
      const res = await api('/add_user_stroke', {
        type: 'text',
        text,
        point: [x, y],
        text_color: $('#textColor')?.value || '#0066ff',
        text_size: parseInt($('#textSize')?.value || '16', 10)
      });
      console.log('add_user_stroke(text):', res);
      if (res.ok) {
        state.fullSVG = res.full_svg || state.fullSVG;   // <-- REFRESH SVG
        state.isDrawingMode = false;
        state.drawingType = null;
        state.drawStart = null;
        $('#textInput').style.display = 'none';
        const pc = $('#previewContainer'); if (pc) pc.innerHTML = '';
        updateDisplay();                                 // <-- RE-RENDER NOW
      }
    }
    
    
  


    
    // Upload
    $('#uploadForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const fd = new FormData($('#uploadForm'));
      const r = await fetch('/upload', {method: 'POST', body: fd});
      const res = await r.json();
      if (res.ok) {
        state.dataURL = res.data_url;
        state.fullSVG = '';
        state.pendingBlock = null;
        state.queueRemaining = 0;
        state.canUndo = false;
        $('#srcImg').src = state.dataURL;
        
        if (res.grid_info) {
          const gi = res.grid_info;
          state.gridPx = gi.grid_px || state.gridPx;        // <— ADD
          $('#gridInfo').innerHTML = `Grid: ${gi.res_x}×${gi.res_y} cells (${gi.cell_size}px each)`;
          $('#gridInfo').style.display = 'block';
        }

        
        $('#srcImg').onload = () => {
          updateDisplay();
        };
      }
    });
    
    // Task change
    $('#task').addEventListener('change', async () => {
      await api('/set-task', {
        task: $('#task').value,
        prompt: $('#customPrompt').value
      });
    });
    
    $('#customPrompt').addEventListener('change', async () => {
      await api('/set-task', {
        task: $('#task').value,
        prompt: $('#customPrompt').value
      });
    });
    
    // Mode change
    $('#mode').addEventListener('change', async () => {
      state.mode = $('#mode').value;
      await api('/set-mode', {mode: state.mode});
      updateDisplay();
    });
    
    // Grid toggle
    $('#showGrid').addEventListener('change', async () => {
      state.showGrid = $('#showGrid').checked;
      const res = await api('/set-grid', {grid: state.showGrid});
      if (res.ok && res.data_url) {
        state.dataURL = res.data_url;
        $('#srcImg').src = state.dataURL;
      }
      updateDisplay();
    });
    
    // Call model
    $('#btnCall').addEventListener('click', async () => {
      $('#btnCall').disabled = true;
      $('#btnCall').textContent = 'Calling model...';
      
      const res = await api('/call_model', {});
      
      
       if (res.accepted_xml_text !== undefined) {
        state.acceptedXmlText = res.accepted_xml_text;
        syncAcceptedXmlUI();
      }
      
      $('#btnCall').disabled = false;
      $('#btnCall').textContent = 'Call Model';
      
      if (res.ok) {
        console.log('Model response:', res);
        if (res.placed_all) {
          // One-turn mode
          state.fullSVG = res.full_svg;
          state.pendingBlock = null;
          state.pendingPreview = null;
          state.queueRemaining = 0;
          state.canUndo = true;
          console.log('One-turn mode: all strokes placed');
        } else {
          // Multi-turn mode
          state.pendingBlock = res.pending;
          state.pendingPreview = res.pending_preview || null;
          state.queueRemaining = res.queue_length - 1;
          state.isDraggingPending = false;
          console.log('Multi-turn mode: pending preview set', state.pendingPreview ? 'YES' : 'NO');
          console.log('Pending block:', state.pendingBlock);
        }
        
        if (res.final_answer) {
          $('#finalAnswer').textContent = res.final_answer;
          $('#answerRow').style.display = 'block';
        }
  
  
        updateDisplay();
        await loadDebugInfo();
      } else {
        console.error('Model call failed:', res.error);
      }
    });
    
    // Accept stroke
    $('#btnAccept').addEventListener('click', async () => {
      const res = await api('/accept_stroke', {});
      if (res.ok) {
        if (res.accepted_svg) {
          const tempDiv = document.createElement('div');
          tempDiv.innerHTML = state.fullSVG || '<svg></svg>';
          const svg = tempDiv.querySelector('svg');
          if (svg) {
            svg.innerHTML += res.accepted_svg;
            state.fullSVG = tempDiv.innerHTML;
          }
        }
        state.pendingBlock = res.pending;
        state.pendingPreview = res.pending_preview || null;
        state.queueRemaining = res.queue_remaining;
        state.canUndo = res.can_undo || false;
        updateDisplay();
      }
    });
    
    
    // Add own stroke
    $('#btnAddOwn').addEventListener('click', () => {
      $('#drawingControls').classList.toggle('active');
    });
    
    // Free Draw tool button
    $('#btnFreeDraw')?.addEventListener('click', () => {
      setMove(false);                 // always disable Move
      state.isDrawingMode = true;
      state.drawingType  = 'freehand';
      state.drawStart    = null;
      $('#drawingControls')?.classList.add('active');
      $('#textInput')?.style?.setProperty('display','none');
      updateDisplay();
    });

    
    $('#btnDrawLine').addEventListener('click', () => {
      setMove(false); // disable move mode
      state.isDrawingMode = true;
      state.drawingType = 'line';
      state.drawStart = null;
      updateDisplay(); // ensures .interactive is applied to overlay
    });

    $('#btnAddText').addEventListener('click', () => {
      setMove(false); // disable move mode
      state.isDrawingMode = true;
      state.drawingType = 'text';
      state.drawStart = null;
      $('#textInput').style.display = 'none';
      updateDisplay();
    });

    
    $('#btnCancelDraw').addEventListener('click', () => {
      state.isDrawingMode = false;
      state.drawingType = null;
      state.drawStart = null;
      $('#drawingControls').classList.remove('active');
      $('#textInput').style.display = 'none';
      updateDisplay();
    });
    
    $('#btnPlaceText').addEventListener('click', async () => {
      const text = $('#textValue').value.trim();
      if (!text || !state.drawStart) return;
      
      const res = await api('/add_user_stroke', {
        type: 'text',
        text: text,
        point: [state.drawStart.x, state.drawStart.y],
        text_color: $('#textColor')?.value || '#0066ff',
        text_size: parseInt($('#textSize')?.value || '16', 10)
      });

      
      if (res.ok) {
        // Use the full SVG returned from backend
        state.fullSVG = res.full_svg;
        state.canUndo = true;
      }
      
      state.isDrawingMode = false;
      state.drawingType = null;
      state.drawStart = null;
      $('#textValue').value = '';
      $('#textInput').style.display = 'none';
      $('#drawingControls').classList.remove('active');
      updateDisplay();
    });
    
    // Accept all remaining
    $('#btnAcceptAll').addEventListener('click', async () => {
      $('#btnAcceptAll').disabled = true;
      $('#btnAcceptAll').textContent = 'Accepting...';
      
      const res = await api('/accept_all', {});
      
      $('#btnAcceptAll').disabled = false;
      $('#btnAcceptAll').textContent = 'Accept All Remaining';
      
      if (res.ok) {
        // Add all accepted SVGs to display
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = state.fullSVG || '<svg></svg>';
        const svg = tempDiv.querySelector('svg');
        if (svg && res.all_svg) {
          for (const svgPart of res.all_svg) {
            svg.innerHTML += svgPart;
          }
          state.fullSVG = tempDiv.innerHTML;
        }
        state.pendingBlock = null;
        state.queueRemaining = 0;
        state.canUndo = res.can_undo || false;
        updateDisplay();
      }
    });
    
    // Request new call with context
    $('#btnRequestNew').addEventListener('click', async () => {
      $('#btnRequestNew').disabled = true;
      $('#btnRequestNew').textContent = 'Requesting...';

      // clear any stale pending preview immediately so the canvas looks empty while we wait
      state.pendingBlock = null;
      state.pendingPreview = null;
      state.isDraggingPending = false;
      updateDisplay();

      const res = await api('/call_model_with_context', {});
      
      
       if (res.accepted_xml_text !== undefined) {
        state.acceptedXmlText = res.accepted_xml_text;
        syncAcceptedXmlUI();
      }

      $('#btnRequestNew').disabled = false;
      $('#btnRequestNew').textContent = 'Request New Call';

      if (res.ok) {
        // hydrate BOTH pending block and its preview from server
        state.pendingBlock    = res.pending || null;
        state.pendingPreview  = res.pending_preview || null;
        state.queueRemaining  = Math.max(0, (res.queue_length || 0) - 1);
        state.isDraggingPending = false;

        // also cancel any in-progress user drawing so the Accept/Move buttons are usable
        state.isDrawingMode = false;
        state.drawingType = null;
        state.drawStart = null;

        // (optional) show final answer if your server returns it
        if (res.final_answer) {
          $('#finalAnswer').textContent = res.final_answer;
          $('#answerRow').style.display = 'block';
        }

        updateDisplay();
        await loadDebugInfo();
      }
    });

    
    // Undo
    $('#btnUndo').addEventListener('click', async () => {
      const res = await api('/undo', {});
      if (res.ok) {
        state.fullSVG = res.full_svg;
        state.canUndo = res.can_undo || false;
        updateDisplay();
      }
    });
    
    // Download
    $('#btnDownload').addEventListener('click', async () => {
      const res = await api('/download_annotated', {});
      if (res.ok) {
        alert(res.message || 'Right-click the image and select "Save Image As..." to download');
      }
    });
    
    // Reset
    $('#btnDone').addEventListener('click', async () => {
      const res = await api('/done', {});
      if (res.ok) {
        state.fullSVG = '';
        state.pendingBlock = null;
        state.queueRemaining = 0;
        state.canUndo = false;
        if (res.data_url) {
          state.dataURL = res.data_url;
          $('#srcImg').src = state.dataURL;
        }
        updateDisplay();
      }
    });
    
    // Debug toggle
    $('#debugToggle').addEventListener('click', () => {
      const panel = $('#debugPanel');
      if (panel.style.display === 'none') {
        panel.style.display = 'block';
        loadDebugInfo();
      } else {
        panel.style.display = 'none';
      }
    });
    
    
    // --- FREEHAND (draw only when in FREEHAND mode; no separate flag) ---
    let freePath = [];
    let freePreview = null;
    let freeInFlight = false;

    function ensurePreviewLayer() {
      return document.querySelector('#svgContainer svg');
    }
    
    function svgPointFromClient(evt, svgEl) {
      const pt = svgEl.createSVGPoint();
      pt.x = evt.clientX; pt.y = evt.clientY;
      const p = pt.matrixTransform(svgEl.getScreenCTM().inverse());
      return { x: p.x, y: p.y };
    }

    
    
    function svgPointFromEvent(evt, svgEl) {
      const pt = svgEl.createSVGPoint();
      pt.x = evt.clientX; pt.y = evt.clientY;
      const p = pt.matrixTransform(svgEl.getScreenCTM().inverse());
      return { x: p.x, y: p.y };
    }
    function previewColor(){ return (document.getElementById('lineColor')?.value || '#00ff00'); }
    function previewWidth(){ return parseFloat(document.getElementById('lineWidth')?.value || '2'); }

    // START stroke
    document.getElementById('svgContainer').addEventListener('pointerdown', (e) => {
      if (!state.isDrawingMode || state.drawingType !== 'freehand') return;
      const mainSvg = ensurePreviewLayer();
      if (!mainSvg) return;

      freePath = [];
      const p = svgPointFromEvent(e, mainSvg);
      freePath.push({ x: p.x, y: p.y, timestamp: performance.now() });

      freePreview = document.createElementNS('http://www.w3.org/2000/svg','polyline');
      freePreview.setAttribute('points', `${p.x},${p.y}`);
      freePreview.setAttribute('fill', 'none');
      freePreview.setAttribute('stroke', previewColor());
      freePreview.setAttribute('stroke-width', previewWidth());
      freePreview.setAttribute('stroke-linecap', 'round');
      freePreview.setAttribute('stroke-linejoin', 'round');
      freePreview.style.pointerEvents = 'none';
      mainSvg.appendChild(freePreview);

      mainSvg.setPointerCapture?.(e.pointerId);
    });

    // DRAW stroke
    document.getElementById('svgContainer').addEventListener('pointermove', (e) => {
      if (!state.isDrawingMode || state.drawingType !== 'freehand' || !freePath.length || !freePreview) return;
      const mainSvg = ensurePreviewLayer();
      if (!mainSvg) return;

      const p = svgPointFromEvent(e, mainSvg);
      freePath.push({ x: p.x, y: p.y, timestamp: performance.now() });
      freePreview.setAttribute('points', freePath.map(pt => `${pt.x},${pt.y}`).join(' '));
    });

    // FINISH stroke (one funnel)
    async function finalizeFreehand() {
      if (!state.isDrawingMode || state.drawingType !== 'freehand' || !freePath.length || freeInFlight) return;
      freeInFlight = true;

      const localPreview = freePreview;
      freePreview = null;
      const payload = freePath.slice();
      freePath = [];

      try {
        const resp = await fetch('/add_freehand', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({
            path: payload,
            line_color: previewColor(),
            line_width: previewWidth()
          })
        }).then(r => r.json());

        if (localPreview && localPreview.parentNode) {
          localPreview.parentNode.removeChild(localPreview);
        }

        const dbg = document.getElementById('debugContent');
        if (resp.ok) {
          if (dbg) {
            let block = '';
            if (resp.xml_block)     block += `<h4>Freehand XML</h4><pre>${escapeHtml(resp.xml_block)}</pre>`;
            if (resp.rendered_svg)  block += `<h4>Rendered SVG (inner)</h4><pre>${escapeHtml(resp.rendered_svg)}</pre>`;
            if (block) dbg.innerHTML = block + (dbg.innerHTML || '');
          }
          state.fullSVG = resp.full_svg;
          state.canUndo = !!resp.can_undo;
          updateDisplay();
        } else {
          console.error('freehand error', resp.error);
          if (dbg) dbg.innerHTML = `<h4 style="color:red;">Freehand Error</h4><pre>${escapeHtml(String(resp.error))}</pre>` + (dbg.innerHTML || '');
        }
      } catch (err) {
        console.error('freehand network error', err);
      } finally {
        freeInFlight = false;
      }
    }

    // Hook all “end” conditions to finalize
    ['pointerup','pointerleave','pointercancel'].forEach(ev =>
      document.getElementById('svgContainer').addEventListener(ev, finalizeFreehand)
    );




    
    // Window resize
    window.addEventListener('resize', updateDisplay);
    $('#srcImg').addEventListener('load', updateDisplay);
    
    // Initialize
    setupDrawingHandlers();
   
    loadSamples();
  </script>
</body>
</html>
"""#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from flask import Flask, jsonify, make_response, request
from PIL import Image

# Import your existing components
try:
    from grid_manager import GridManager
    from llm_adapters import make_adapter, GeminiAdapter
    import prompts
    import utils
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure grid_manager.py, llm_adapters.py, prompts.py, and utils.py are available")
    raise

app = Flask(__name__)

# In-memory session store keyed by cookie 'sid'
SESSIONS: Dict[str, Dict[str, Any]] = {}

# Global configuration (set from command-line args)
CONFIG = {
    "llm": "gemini",
    "model": "gemini-2.5-pro",
    "cell_size": 15,
    "res": 50,
    "max_tokens": 8192,
    "show_full_grid": False,  # Important: should be False for axis-only style
    "adaptive_grid": True,     # Enable adaptive grid by default
    "target_cols": 50,
    "target_rows": 50,
    "min_cell_px": 20,
    "max_cell_px": 64,
}

# ------------- Utilities -------------
import re, math
from typing import Dict, List, Tuple, Optional

def _parse_group_translate(svg_fragment: str) -> Tuple[float, float]:
    """
    Extract translate(tx,ty) from the outer <g ...> tag if present.
    Only handles a single translate(...) (which is how our code writes it).
    """
    m = re.search(r'<g[^>]*\btransform="[^"]*translate\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', svg_fragment)
    if not m: 
        return 0.0, 0.0
    return float(m.group(1)), float(m.group(2))

def _extract_visible_line(svg_fragment: str) -> Optional[Tuple[float,float,float,float]]:
    """
    Return (x1,y1,x2,y2) for the visible line in our user stroke group.
    We skip the transparent halo line. If multiple <line>, pick the last non-transparent.
    """
    lines = re.findall(r'<line\b([^>]*)/>', svg_fragment, re.S)
    if not lines:
        return None
    chosen = None
    for attrs in lines:
        # Ignore transparent halo
        if re.search(r'stroke\s*=\s*["\']transparent["\']', attrs, re.I):
            continue
        chosen = attrs
    if not chosen:
        return None
    def _get(name):
        m = re.search(fr'\b{name}\s*=\s*["\']([-\d.]+)["\']', chosen)
        return float(m.group(1)) if m else None
    x1 = _get('x1'); y1 = _get('y1'); x2 = _get('x2'); y2 = _get('y2')
    if None in (x1,y1,x2,y2):
        return None
    return x1,y1,x2,y2

def _extract_text(svg_fragment: str) -> Optional[Tuple[float,float,str]]:
    """
    Return (x,y,text) for the <text> element inside our user group.
    """
    m = re.search(r'<text\b[^>]*\bx\s*=\s*["\']([-\d.]+)["\'][^>]*\by\s*=\s*["\']([-\d.]+)["\'][^>]*>(.*?)</text>', svg_fragment, re.S)
    if not m:
        return None
    x = float(m.group(1)); y = float(m.group(2))
    # Strip any nested tags; we expect plain text
    txt = re.sub(r'<[^>]+>', '', m.group(3)).strip()
    return x, y, txt

def _nearest_grid_label(px: float, py: float, grid_mgr) -> str:
    """
    Map pixel (px,py) to the nearest grid key like 'x21y34', using GridManager.positions.
    This is O(G) over grid points; fine for ~50x50. Cache list in session if needed.
    """
    best_key, best_d2 = None, float('inf')
    for key, (gx, gy) in grid_mgr.positions.items():
        d2 = (gx - px)*(gx - px) + (gy - py)*(gy - py)
        if d2 < best_d2:
            best_key, best_d2 = key, d2
    return best_key or "x0y0"

def _user_svg_to_blocks(svg_fragment: str, grid_mgr) -> List[str]:
    """
    Convert our stored user-stroke SVG fragment (wrapped in <g id="user_N"...>) into one or more
    XML stroke blocks in the same format we ask the model to output.
    Handles: line, text. (Rects/paths can be added later similarly.)
    """
    blocks: List[str] = []
    tx, ty = _parse_group_translate(svg_fragment)

    # LINE
    line = _extract_visible_line(svg_fragment)
    if line:
        x1,y1,x2,y2 = line
        x1 += tx; y1 += ty; x2 += tx; y2 += ty
        p1 = _nearest_grid_label(x1, y1, grid_mgr)  # e.g., 'x22y33'
        p2 = _nearest_grid_label(x2, y2, grid_mgr)
        blocks.append(
            "<s1>\n"
            f"  <points>'{p1}','{p2}'</points>\n"
            "  <t_values>0.00,1.00</t_values>\n"
            "</s1>"
        )

    # TEXT
    t = _extract_text(svg_fragment)
    if t:
        x,y,txt = t
        x += tx; y += ty
        p = _nearest_grid_label(x, y, grid_mgr)
        # follow your text-stroke schema
        blocks.append(
            "<s_text>\n"
            f"  <text>'{txt}'</text>\n"
            f"  <points>'{p}'</points>\n"
            "</s_text>"
        )

    return blocks


def _strip_outer_svg(svg_str: str) -> str:
    """If svg_str is <svg ...>inner</svg>, return just inner; else return as-is."""
    m = re.match(r'^\s*<svg[^>]*>(.*)</svg>\s*$', svg_str, re.S)
    return m.group(1) if m else svg_str




def _ensure_session() -> Tuple[str, Dict[str, Any], bool]:
    """Create or fetch a per-user session; returns (sid, session_dict, is_new)."""
    sid = request.cookies.get("sid", "")
    new_sid = False
    if not sid or sid not in SESSIONS:
        sid = str(uuid.uuid4())
        new_sid = True
        
        # Initialize GridManager with configured settings (ADAPTIVE GRID)
        grid_mgr = GridManager(
            cell_size=CONFIG["cell_size"],
            min_grid=10,
            max_grid=100,
            adaptive_grid=CONFIG["adaptive_grid"],    # Use adaptive grid
            target_cols=CONFIG["target_cols"],
            target_rows=CONFIG["target_rows"],
            min_cell_px=CONFIG["min_cell_px"],
            max_cell_px=CONFIG["max_cell_px"],
        )
        
        SESSIONS[sid] = {
            "task": "counting",
            "prompt": "",
            "image_b64": None,
            "image_mime": None,
            "canvas_with_grid_b64": None,  # Image with grid overlay (what model sees)
            "accepted_strokes": [],
            "pending_queue": [],
            "pending": None,
            "grid": True,
            "mode": "multi_turn",  # multi_turn | one_turn
            "grid_manager": grid_mgr,
            "stroke_counter": 0,
            "all_strokes_svg": "",
            "stroke_history": [],  # For undo
            "debug_info": None,  # Store last API call details
            "meta": {},
            "style": {
                "stroke_color": "#111111",
                "stroke_width": 2.0,
                "preview_color": "#f97316",
                "dash": False,
                "text_color": "#0066ff",
                "text_size": 16
            },
            "user_xml_strokes": [], 
        }
    return sid, SESSIONS[sid], new_sid

def _img_to_b64(file_storage) -> Tuple[str, str]:
    data = file_storage.read()
    mime = file_storage.mimetype or "image/png"
    b64 = base64.b64encode(data).decode("utf-8")
    return b64, mime

def _build_system_prompt(grid_mgr: GridManager) -> str:
    """Use the actual system prompt from prompts.py with current grid dimensions."""
    return prompts.system_prompt.format(
        res_x=grid_mgr.res_x,
        res_y=grid_mgr.res_y
    )

def _build_user_prompt(task: str, custom_prompt: str) -> str:
    """Build user prompt based on task type using actual prompts from prompts.py."""
    if task == "counting":
        thing = custom_prompt if custom_prompt else "object"
        # Fix: COUNTING_PROMPT uses {thing} not {object}
        return prompts.COUNTING_PROMPT.format(object=thing)
    elif task == "labeling":
        concept = custom_prompt if custom_prompt else "object"
        labels_hint = "visible parts"
        return prompts.GENERIC_LABEL_PROMPT.format(
            concept=concept,
            labels_hint=labels_hint
        )
    else:
        # For custom tasks, use the toolkit
        return f"{custom_prompt}\n\n{prompts.MIX_TOOLKIT}"


def _dedupe_consecutive_tokens_and_t(tokens, tvals=None):
    """
    tokens: ["'x35y18'","'x35y18'", "'x35y17'", ...]
    tvals : ignored; we recompute a clean linear schedule 0..1

    Returns (new_tokens, new_tvals)
    """
    if not tokens:
        return [], []

    new_tokens = [tokens[0]]
    for tok in tokens[1:]:
        if tok != new_tokens[-1]:
            new_tokens.append(tok)

    # Ensure at least 2 points so the stroke isn't degenerate
    if len(new_tokens) == 1:
        new_tokens = [new_tokens[0], new_tokens[0]]

    n = len(new_tokens)
    new_tvals = [f"{i/(n-1):.2f}" for i in range(n)]
    return new_tokens, new_tvals

      
def _accepted_strokes_text(sess: Dict) -> str:
    """
    Return a text block list of ALL accepted strokes in the same XML format the model outputs.
    For model strokes, we already have XML 'block'. For user strokes (no 'block'), convert their SVG.
    """
    grid_mgr = sess.get("grid_manager")
    parts = []
    for h in sess.get("stroke_history", []):
        if h.get("block"):
            parts.append(str(h["block"]).strip())
        else:
            svg = (h.get("svg") or "").strip()
            if svg and grid_mgr:
                try:
                    user_blocks = _user_svg_to_blocks(svg, grid_mgr)
                    parts.extend(user_blocks)
                except Exception as _:
                    # Fallback: include raw SVG if conversion failed (rare)
                    parts.append(svg)
    return "\n".join(parts)



def _parse_llm_strokes(answer_xml: str, grid_mgr: GridManager) -> List[str]:
    """
    Parse stroke blocks from LLM output using the same parsing as collab_sketch.
    Returns list of stroke block strings (e.g., '<s1>...</s1>').
    """
    # Normalize the output
    answer_xml = re.sub(r"^```(?:xml|html)?\s*|\s*```$", "", answer_xml.strip())
    
    # Extract all <sN>...</sN> blocks
    blocks = re.findall(r"(<s\d+>.*?</s\d+>)", answer_xml, re.S)
    return blocks


def _wrap_svg(fragment: str, grid_mgr) -> str:
    """Return a full <svg> wrapper for preview layers."""
    w, h = getattr(grid_mgr, "grid_size", (1024, 1024))
    return f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">{fragment}</svg>'

def _apply_dash(svg: str, dash: bool) -> str:
    """Optionally add a dashed stroke to the first stroked element."""
    if not dash or not svg:
        return svg
    return re.sub(r'(stroke="[^"]+")', r'\1 stroke-dasharray="6 6"', svg, count=1)

def _px_to_grid_tokens(path_px, grid_mgr):
    """
    path_px: list of dicts/tuples with (x,y) in canvas pixel coords.
    Returns (tokens:list[str], tvals:list[str]).
    """
    pos_items = list(grid_mgr.positions.items())
    n = max(2, len(path_px))
    tvals = [f"{i/(n-1):.2f}" for i in range(n)]

    tokens = []
    for pt in path_px:
        x, y = (pt["x"], pt["y"]) if isinstance(pt, dict) else pt
        best_tok, best_d2 = None, 1e18
        for tok, (cx, cy) in pos_items:
            dx = cx - x; dy = cy - y
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2, best_tok = d2, tok
        # turn "x21y34" into "'x21y34'"
        tokens.append(best_tok.replace("x", "'x").replace("y", "y") + "'")
    return tokens, tvals




def _stroke_block_to_svg(
    block: str,
    stroke_no: int,
    grid_mgr: GridManager,
    stroke_color: str = "black",
    stroke_width: float = 2.0,
    dash: bool = False,
) -> str:
    """
    Convert a single stroke block to SVG using the same logic as collab_sketch.
    Handles text strokes, rectangles, two-point lines, and curves.
    Returns a full <svg> for ad-hoc line/rect/polyline so the preview layer can render.
    """
    if not block:
        print("DEBUG _stroke_block_to_svg: block is None or empty")
        return ""
    if not isinstance(block, str):
        print(f"DEBUG _stroke_block_to_svg: block is not a string, type={type(block)}")
        return ""

    print(f"DEBUG _stroke_block_to_svg: Processing block (first 150 chars): {block[:150]}")

    # Normalize short decimals inside the XML
    block = re.sub(
        r'(?<=,|\>)\s*([01])\.([0-9])(?![0-9])',
        lambda m: f"{m.group(1)}.{m.group(2)}0",
        block
    )

    # Extract ID for labeling
    m_id = re.search(r"<id>(.*?)</id>", block, re.S)
    stroke_label = (m_id.group(1).strip() if m_id else f"s{stroke_no}")
    stroke_label = re.sub(r"[^\w\-]", "_", stroke_label)

    # Text stroke
    m_text = re.search(r"<text([^>]*)>\s*'([^']+)'\s*</text>", block, re.S)
    if m_text:
        m_ptblk = re.search(r"<points>(.*?)</points>", block, re.S)
        if not m_ptblk:
            return ""
        pts = re.findall(r"'?x(\d+)y(\d+)'?", m_ptblk.group(1))
        if not pts:
            return ""
        gx, gy = pts[0]
        key = f"x{int(gx)}y{int(gy)}"
        if key not in grid_mgr.positions:
            return ""
        cx, cy = grid_mgr.positions[key]
        text_val = m_text.group(2)
        font_px = int(max(10, grid_mgr.cell_size * 3.2))  # same default you used

        frag = (
            f'<g id="{stroke_label}_s{stroke_no}">'
            f'<text x="{cx:.1f}" y="{cy:.1f}" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'font-family="Arial" font-size="{font_px}" fill="{stroke_color}">'
            f'{text_val}</text></g>'
        )
      
        return _wrap_svg(frag, grid_mgr)

    # Points block (line/rect/polyline)
    m_ptblk = re.search(r"<points>(.*?)</points>", block, re.S)
    if m_ptblk:
        grid_pts = [(int(gx), int(gy)) for gx, gy in re.findall(r"x(\d+)y(\d+)", m_ptblk.group(1))]

        # Two-point line (this was your failing case)
        if len(grid_pts) == 2:
            (gx1, gy1), (gx2, gy2) = grid_pts
            k1 = f"x{gx1}y{gy1}"
            k2 = f"x{gx2}y{gy2}"
            if k1 in grid_mgr.positions and k2 in grid_mgr.positions:
                (x1, y1) = grid_mgr.positions[k1]
                (x2, y2) = grid_mgr.positions[k2]
                frag = (
                    f'<g id="{stroke_label}_s{stroke_no}">'
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none"/>'
                    f'</g>'
                )
                frag = _apply_dash(frag, dash)
                return _wrap_svg(frag, grid_mgr)

        # Axis-aligned rectangle from 4 corners
        if len(grid_pts) >= 4:
            px_pts = []
            for gx, gy in grid_pts:
                key = f"x{gx}y{gy}"
                if key in grid_mgr.positions:
                    px_pts.append(grid_mgr.positions[key])
            if len(px_pts) >= 4:
                closed = (px_pts[0] == px_pts[-1])
                uniq = px_pts[:-1] if closed else px_pts
                uniq_corners = list(dict.fromkeys(uniq))
                if len(uniq_corners) == 4:
                    xs = sorted({x for x, _ in uniq_corners})
                    ys = sorted({y for _, y in uniq_corners})
                    if len(xs) == 2 and len(ys) == 2:
                        x0, x1 = xs
                        y0, y1 = ys
                        x, y = min(x0, x1), min(y0, y1)
                        w, h = abs(x1 - x0), abs(y1 - y0)
                        w = w if w > 0 else 1
                        h = h if h > 0 else 1
                        frag = (
                            f'<g id="{stroke_label}_s{stroke_no}">'
                            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                            f'stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none"/>'
                            f'</g>'
                        )
                        frag = _apply_dash(frag, dash)
                        return _wrap_svg(frag, grid_mgr)

    # Default: curves/paths via your utils (usually already returns a full <svg>)
    try:
        strokes_list_str, t_values_str = utils.parse_xml_string_single_stroke(
            block, grid_mgr.res_x, stroke_no, grid_mgr.res_x, grid_mgr.res_y
        )
        import ast
        strokes_list = ast.literal_eval(strokes_list_str)
        t_values = ast.literal_eval(t_values_str)

        if len(t_values) != len(strokes_list):
            n = len(strokes_list)
            t_values = [0.00] * n if n <= 1 else [round(i/(n-1), 2) for i in range(n)]

        all_control_points = utils.get_control_points_single_stroke(
            strokes_list, t_values, grid_mgr.positions
        )
        return utils.format_svg_single_stroke(
            all_control_points,
            dim=grid_mgr.grid_size,
            stroke_width=stroke_width if stroke_width else max(1.0, grid_mgr.cell_size * 0.6),
            stroke_counter=stroke_no,
            group_id=stroke_label,
            stroke_color=stroke_color
        )
    except Exception as e:
        # Fallback: polyline from any points we can parse → ensure full <svg>
        try:
            pts = re.findall(r"x(\d+)y(\d+)", m_ptblk.group(1)) if m_ptblk else []
            px_pts = []
            for gx, gy in [(int(a), int(b)) for a, b in pts]:
                key = f"x{gx}y{gy}"
                if key in grid_mgr.positions:
                    px_pts.append(grid_mgr.positions[key])
            if len(px_pts) >= 2:
                path = " ".join(f"{x:.1f},{y:.1f}" for (x, y) in px_pts)
                frag = (
                    f'<g id="{stroke_label}_s{stroke_no}">'
                    f'<polyline points="{path}" stroke="{stroke_color}" stroke-width="{stroke_width}" fill="none" />'
                    f'</g>'
                )
                frag = _apply_dash(frag, dash)
                return _wrap_svg(frag, grid_mgr)
        except Exception:
            pass
        print(f"Error parsing stroke: {e}")
        return ""
      
      
def _extract_final_answer(answer: str) -> Optional[str]:
    if not answer:
        return None
    m = re.search(r"<final_answer>\s*(.*?)\s*</final_answer>", answer, re.S | re.I)
    return m.group(1).strip() if m else None



def _get_adapter():
    """Get configured LLM adapter."""
    return make_adapter(
        CONFIG["llm"],
        CONFIG["model"],
        cache=False,
        max_tokens=CONFIG["max_tokens"]
    )

def _create_canvas_with_grid(img: Image.Image, grid_mgr: GridManager, show_grid: bool = True) -> Tuple[Image.Image, str]:
    """
    Create the canvas that will be sent to the model (image + optional grid overlay).
    Returns: (PIL Image, base64 string)
    """
    if not show_grid:
        canvas = img.copy()
    else:
        canvas = grid_mgr.create_annotated_image(
            img,
            show_full_grid=CONFIG["show_full_grid"],  # False = axis-only
            bgcolor=(255, 255, 255),
        )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return canvas, b64



def _compose_pending_svg(sess) -> str:
    base = sess.get("pending_svg_base")
    if not base:
        return ""
    dx = float(sess.get("pending_dx", 0.0) or 0.0)
    dy = float(sess.get("pending_dy", 0.0) or 0.0)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return base

    # If base is a full <svg>, wrap its inner content in a translated <g>
    m = re.match(r'^<svg([^>]*)>(.*)</svg>\s*$', base, re.S)
    if m:
        attrs, inner = m.group(1), m.group(2)
        return f'<svg{attrs}><g transform="translate({dx}, {dy})">{inner}</g></svg>'

    # Otherwise, just wrap the fragment in a translated <g> and an <svg> shell
    gm = sess["grid_manager"]
    w, h = gm.grid_size
    return f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg"><g transform="translate({dx}, {dy})">{base}</g></svg>'


def _composite_canvas_with_svg(sess: Dict) -> Optional[str]:
    """Return PNG base64 of canvas_with_grid + accepted all_strokes_svg overlaid.
       Uses cairosvg if available; falls back to base canvas if unavailable."""
    try:
        import cairosvg  # type: ignore
    except Exception:
        return sess.get("canvas_with_grid_b64")  # fallback

    canvas_b64 = sess.get("canvas_with_grid_b64")
    svg = sess.get("all_strokes_svg", "")
    if not canvas_b64:
        return None
    try:
        # Decode base canvas
        base = Image.open(io.BytesIO(base64.b64decode(canvas_b64))).convert("RGBA")
        w, h = base.size
        # Render SVG to PNG same size
        png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'), output_width=w, output_height=h)
        overlay = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        # Composite
        base.alpha_composite(overlay)
        out = io.BytesIO()
        base.save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode("utf-8")
    except Exception:
        return canvas_b64



def _freehand_points_to_xml(sess, stroke_points):
    """
    stroke_points: [{"x": px, "y": px, "timestamp": ms}, ...]
    Returns an <sN>...</sN> block. More tolerant snapping + fallback.
    """
    gm = sess["grid_manager"]
    cell = gm.cell_size
    W, H = gm.grid_size
    res_x, res_y = gm.res_x, gm.res_y
    positions = gm.positions  # 'x{i}y{j}' -> (cx, cy)

    # Tolerances scale with cell size
    tol_strict = max(5.0, 0.40 * cell)   # keep if within 40% cell
    tol_loose  = max(8.0, 0.65 * cell)   # fallback to 65% cell

    def nearest_token(x, y):
        gx = int(max(0, min(res_x - 1, x // cell)))
        gy = int(max(0, min(res_y - 1, (H - 1 - y) // cell)))
        tok = f"x{gx}y{gy}"
        if tok in positions:
            cx, cy = positions[tok]
            d = ((x - cx)**2 + (y - cy)**2) ** 0.5
            return tok, d
        return None, None

    tokens = []
    tvals  = []

    # First pass (strict)
    for p in stroke_points:
        tok, d = nearest_token(p["x"], p["y"])
        if tok and d <= tol_strict and (not tokens or tokens[-1] != tok):
            tokens.append(tok)
            tvals.append(p["timestamp"])

    # Second pass (loose) if nothing
    if not tvals:
        for p in stroke_points:
            tok, d = nearest_token(p["x"], p["y"])
            if tok and d <= tol_loose and (not tokens or tokens[-1] != tok):
                tokens.append(tok)
                tvals.append(p["timestamp"])

    # Final fallback: always build something by force-mapping to nearest cells
    if not tvals:
        for p in stroke_points:
            tok, _ = nearest_token(p["x"], p["y"])
            if tok and (not tokens or tokens[-1] != tok):
                tokens.append(tok)
                tvals.append(p["timestamp"])

    # If still nothing, give up
    if not tvals:
        raise ValueError("No usable points from freehand stroke.")

    # Normalize t-values
    t0, t1 = min(tvals), max(tvals)
    if t1 == t0:
        norm = ["0.00"] * len(tvals)
    else:
        norm = [f"{(t - t0) / (t1 - t0):.2f}" for t in tvals]

    s_no = sess["stroke_counter"] + 1
    points_str = ",".join(f"'{t}'" for t in tokens)
    tvals_str  = ",".join(norm)

    xml = f"<s{s_no}>\n  <points>{points_str}</points>\n  <t_values>{tvals_str}</t_values>\n</s{s_no}>"
    # stash for debugging
    sess.setdefault("debug_info", {})
    sess["debug_info"]["last_freehand_xml"] = xml
    return xml

def _composite_canvas_with_svg(sess):
    """
    Returns PNG base64 of (base image with grid) + overlay SVG rendered on top.
    If cairosvg is not available, return None (UI will fall back to existing canvas).
    """
    try:
        import cairosvg
        from PIL import Image
        import io, base64

        # Base canvas with grid
        if not sess.get("canvas_with_grid_b64"):
            return None
        base_png = base64.b64decode(sess["canvas_with_grid_b64"])
        base_img = Image.open(io.BytesIO(base_png)).convert("RGBA")

        # Render overlay SVG to PNG
        overlay_svg = sess.get("all_strokes_svg", "")
        if not overlay_svg:
            # nothing to composite
            buf = io.BytesIO()
            base_img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

        overlay_png = cairosvg.svg2png(bytestring=overlay_svg.encode("utf-8"))
        overlay_img = Image.open(io.BytesIO(overlay_png)).convert("RGBA")

        # Composite
        base_img.alpha_composite(overlay_img)

        out = io.BytesIO()
        base_img.save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode("utf-8")
    except Exception:
        # no cairosvg or any other issue -> just skip
        return None





def _call_model(sess: Dict, adapter=None) -> List[str]:
    """Call the model and return list of stroke blocks."""
    if not sess.get("canvas_with_grid_b64"):
        return []
    
    if adapter is None:
        adapter = _get_adapter()

    grid_mgr = sess["grid_manager"]
    system_msg = _build_system_prompt(grid_mgr)
    user_msg = _build_user_prompt(sess["task"], sess.get("prompt", ""))
    
    
    accepted_xml_text = _accepted_strokes_text(sess)
    if accepted_xml_text.strip():
        user_msg += (
            "\n\nAccepted strokes so far (do not duplicate these). Look at where these strokes were drawn so you don't repeat:\n"
            + accepted_xml_text
        )
    

    # NEW: prefer composited image (base + SVG overlays). Fallback to grid canvas.
    composited_b64 = _composite_canvas_with_svg(sess)
    img_b64 = composited_b64 or sess.get("canvas_with_grid_b64")
    mime = sess.get("image_mime", "image/png")

    content = [
        {"type": "image",
         "source": {"type": "base64", "media_type": mime, "data": img_b64}},
        {"type": "text", "text": user_msg},
    ]
    messages = [{"role": "user", "content": content}]
    
    # Store debug info
    debug_info = {
        "timestamp": datetime.now().isoformat(),
        "system_prompt": system_msg,
        "user_prompt": user_msg,
        "grid_info": grid_mgr.get_grid_info(),
        "has_image": True,
        "model_output": None,
        "error": None
    }
    
    # Call model
    try:
        use_stop = not isinstance(adapter, GeminiAdapter)
        additional_args = {}
        if use_stop:
            additional_args["stop_sequences"] = "</answer>"
        
        raw = adapter.call(system_msg, messages, additional_args)
        answer = adapter.extract_text(raw)
        final_ans = _extract_final_answer(answer)
        if final_ans:
            sess["final_answer"] = final_ans
        
        debug_info["model_output"] = answer
        debug_info["raw_response_preview"] = str(raw)[:500] if raw else None
        
        # Store debug info in session
        sess["debug_info"] = debug_info
        
        # Parse stroke blocks
        blocks = _parse_llm_strokes(answer, grid_mgr)
        return blocks
    except Exception as e:
        debug_info["error"] = str(e)
        sess["debug_info"] = debug_info
        print(f"Model call error: {e}")
        return []

# ------------- Routes -------------
@app.route("/", methods=["GET"])
def index():
    sid, sess, new_sid = _ensure_session()
    resp = make_response(INDEX_HTML)
    if new_sid:
        resp.set_cookie("sid", sid, httponly=True, samesite="Lax")
    return resp

@app.post("/upload")
def upload():
    sid, sess, _ = _ensure_session()

    # 1) Multipart file?
    f = request.files.get("image") if request.files else None

    # 2) Or JSON data URL?
    image_b64_from_json = None
    show_grid_json = True
    if not f:
        data = request.get_json(silent=True) or {}
        data_url = data.get("image_data")
        show_grid_json = bool(data.get("show_grid", True))
        if isinstance(data_url, str) and data_url.startswith("data:"):
            try:
                header, b64 = data_url.split(",", 1)
                image_b64_from_json = b64
            except Exception:
                pass

    if not f and not image_b64_from_json:
        return jsonify({"ok": False, "error": "no image"}), 400

    # Load image
    if f:
        b64, mime = _img_to_b64(f)
    else:
        b64, mime = image_b64_from_json, "image/png"

    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

    # Update grid for image (adaptive grid honored)
    grid_mgr = sess["grid_manager"]
    grid_mgr.update_grid_for_image(img, CONFIG["show_full_grid"])

    # Compose canvas-with-grid (this is what both the model and the UI should see)
    canvas, canvas_b64 = _create_canvas_with_grid(img, grid_mgr, show_grid_json)

    # Reset session state
    sess["image_b64"] = b64
    sess["image_mime"] = mime
    sess["canvas_with_grid_b64"] = canvas_b64
    sess["accepted_strokes"] = []
    sess["pending_queue"] = []
    sess["pending"] = None
    sess["stroke_counter"] = 0
    sess["all_strokes_svg"] = ""
    sess["stroke_history"] = []
    sess["debug_info"] = None
    sess["user_xml_strokes"] = []  # reset on new image


    info = grid_mgr.get_grid_info()
    info["grid_px"] = {"w": grid_mgr.grid_size[0], "h": grid_mgr.grid_size[1]}  # ADD

    return jsonify({
        "ok": True,
        "data_url": f"data:image/png;base64,{canvas_b64}",
        "grid_info": info  # use 'info' instead of grid_mgr.get_grid_info()
    })


@app.post("/set-task")
def set_task():
    sid, sess, _ = _ensure_session()
    data = request.get_json(silent=True) or {}
    task = str(data.get("task", "counting")).strip().lower()
    sess["task"] = task
    sess["prompt"] = str(data.get("prompt", ""))
    return jsonify({"ok": True, "task": task})

@app.post("/set-mode")
def set_mode():
    sid, sess, _ = _ensure_session()
    data = request.get_json(silent=True) or {}
    mode = str(data.get("mode", "multi_turn")).strip().lower()
    if mode not in ("multi_turn", "one_turn"):
        mode = "multi_turn"
    sess["mode"] = mode
    return jsonify({"ok": True, "mode": mode})

@app.post("/set-grid")
def set_grid():
    sid, sess, _ = _ensure_session()
    data = request.get_json(silent=True) or {}
    sess["grid"] = bool(data.get("grid", True))
    
    # Regenerate canvas if we have an image
    if sess.get("image_b64"):
        img = Image.open(io.BytesIO(base64.b64decode(sess["image_b64"]))).convert("RGB")
        grid_mgr = sess["grid_manager"]
        canvas, canvas_b64 = _create_canvas_with_grid(img, grid_mgr, sess["grid"])
        sess["canvas_with_grid_b64"] = canvas_b64
        return jsonify({
            "ok": True, 
            "grid": sess["grid"],
            "data_url": f"data:image/png;base64,{canvas_b64}"
        })
    
    return jsonify({"ok": True, "grid": sess["grid"]})

@app.post("/grid_overlay")
def grid_overlay():
    """Return the canvas with grid overlay (what model sees)."""
    sid, sess, _ = _ensure_session()
    if not sess.get("canvas_with_grid_b64"):
        return jsonify({"ok": False, "error": "no image"}), 400
    
    return jsonify({
        "ok": True, 
        "data_url": f"data:image/png;base64,{sess['canvas_with_grid_b64']}"
    })

@app.post("/call_model")
def call_model():
    sid, sess, _ = _ensure_session()
    if not sess.get("image_b64"):
        return jsonify({"ok": False, "error": "no image"}), 400
    
    adapter = _get_adapter()
    blocks = _call_model(sess, adapter)
    
    if sess.get("mode") == "one_turn":
        # Place all strokes immediately
        sess["stroke_counter"] = 0
        sess["all_strokes_svg"] = f'<svg width="{sess["grid_manager"].grid_size[0]}" height="{sess["grid_manager"].grid_size[1]}" xmlns="http://www.w3.org/2000/svg">'
        
        applied = []
        for block in blocks:
            sess["stroke_counter"] += 1
            svg = _stroke_block_to_svg(block, sess["stroke_counter"], sess["grid_manager"])
            if svg:
                sess["all_strokes_svg"] += svg
                applied.append({"block": block, "svg": svg})
        
        sess["all_strokes_svg"] += "</svg>"
        sess["pending_queue"] = []
        sess["pending"] = None
        
        return jsonify({
            "ok": True,
            "placed_all": True,
            "applied": applied,
            "full_svg": sess["all_strokes_svg"],
            "final_answer": sess.get("final_answer") 
        })
    else:
        # Multi-turn: queue strokes
        sess["pending_queue"] = blocks
        pending_block = blocks[0] if blocks else None

        pending_preview = None
        if pending_block:
            # Generate and store a base preview SVG for pending
            
            st = {"preview_color": "#f97316", "stroke_width": 2.0, "dash": False}
            svg_prev = _stroke_block_to_svg(
                pending_block,
                999,
                sess["grid_manager"],
                stroke_color=st["preview_color"],
                stroke_width=st["stroke_width"],
                dash=st["dash"],
            )
            sess["pending"] = {"block": pending_block}   # keep metadata server-side
            sess["pending_svg_base"] = svg_prev          # base (untranslated) preview svg
            sess["pending_dx"] = 0.0
            sess["pending_dy"] = 0.0
            pending_preview = svg_prev
        else:
            sess["pending"] = None
            sess["pending_svg_base"] = None
            sess["pending_dx"] = 0.0
            sess["pending_dy"] = 0.0

        return jsonify({
            "ok": True,
            "queue_length": len(blocks),
            # Keep returning the raw block string so your existing UI stays unchanged
            "pending": (pending_block or None),
            "pending_preview": pending_preview,
            "final_answer": sess.get("final_answer")
        })


@app.post("/accept_stroke")
def accept_stroke():
    sid, sess, _ = _ensure_session()
    if not sess.get("pending"):
        return jsonify({"ok": False, "error": "no pending stroke"}), 400

    gm = sess["grid_manager"]

    # Build SVG header if needed
    if not sess.get("all_strokes_svg"):
        sess["all_strokes_svg"] = f'<svg width="{gm.grid_size[0]}" height="{gm.grid_size[1]}" xmlns="http://www.w3.org/2000/svg">'

    # Use composed (possibly moved) pending SVG if available; else recompute with ACCEPTED style
    st = sess.get("style", {})
    accepted_color = st.get("stroke_color", "#111111")
    accepted_width = float(st.get("stroke_width", 2.0))
    accepted_dash  = bool(st.get("dash", False))

    placed_svg = _compose_pending_svg(sess)
    if not placed_svg:
        sess["stroke_counter"] += 1
        # pending might be {"block": "..."} or a raw string
        block = sess["pending"]["block"] if isinstance(sess["pending"], dict) else sess["pending"]
        placed_svg = _stroke_block_to_svg(
            block,
            sess["stroke_counter"],
            gm,
            stroke_color=accepted_color,
            stroke_width=accepted_width,
            dash=accepted_dash,
        )
    else:
        sess["stroke_counter"] += 1

    if placed_svg:
        # Append directly to the open SVG (your original behavior)
        sess["all_strokes_svg"] += placed_svg
        sess.setdefault("stroke_history", []).append({
            "stroke_no": sess["stroke_counter"],
            "block": sess["pending"]["block"] if isinstance(sess["pending"], dict) else sess["pending"],
            "svg": placed_svg
        })

    # Advance queue (drop current if it matches)
    next_block = None
    if sess.get("pending_queue"):
        try:
            if sess["pending_queue"][0] == (sess["pending"]["block"] if isinstance(sess["pending"], dict) else sess["pending"]):
                sess["pending_queue"].pop(0)
        except Exception:
            # be tolerant if shapes differ
            sess["pending_queue"].pop(0)
        next_block = sess["pending_queue"][0] if sess["pending_queue"] else None

    # Set up the new pending preview (use PREVIEW style)
    pending_preview = None
    if next_block:
        preview_color = st.get("preview_color", "#f97316")
        preview_width = float(st.get("stroke_width", 2.0))
        preview_dash  = bool(st.get("dash", False))
        svg_prev = _stroke_block_to_svg(
            next_block,
            999,  # serial for preview only
            gm,
            stroke_color=preview_color,
            stroke_width=preview_width,
            dash=preview_dash,
        )
        sess["pending"] = {"block": next_block}
        sess["pending_svg_base"] = svg_prev
        sess["pending_dx"] = 0.0
        sess["pending_dy"] = 0.0
        pending_preview = svg_prev
    else:
        sess["pending"] = None
        sess["pending_svg_base"] = None
        sess["pending_dx"] = 0.0
        sess["pending_dy"] = 0.0
        # Keep the SVG open; you close it elsewhere when fully done.

    return jsonify({
        "ok": True,
        "accepted_svg": placed_svg,
        "pending": (next_block or None),
        "pending_preview": pending_preview or "",
        "queue_remaining": len(sess.get("pending_queue", [])),
        "full_svg": (sess["all_strokes_svg"] + "</svg>") if not next_block and not sess.get("pending_queue") else sess["all_strokes_svg"],
        "can_undo": len(sess.get("stroke_history", [])) > 0
    })



@app.post("/reject_stroke")
def reject_stroke():
    sid, sess, _ = _ensure_session()
    # Move to next without adding current
    sess["pending"] = sess["pending_queue"].pop(0) if sess["pending_queue"] else None
    return jsonify({
        "ok": True,
        "pending": sess["pending"],
        "queue_remaining": len(sess["pending_queue"])
    })

@app.post("/accept_all")
def accept_all():
    """Accept all remaining strokes in the queue."""
    sid, sess, _ = _ensure_session()
    
    applied = []
    # First accept the pending stroke if there is one
    if sess.get("pending"):
        sess["stroke_counter"] += 1
        svg = _stroke_block_to_svg(sess["pending"], sess["stroke_counter"], sess["grid_manager"])
        if svg:
            sess["all_strokes_svg"] += svg
            sess["stroke_history"].append({
                "stroke_no": sess["stroke_counter"],
                "block": sess["pending"],
                "svg": svg
            })
            applied.append(svg)
    
    # Then accept all queued strokes
    while sess["pending_queue"]:
        sess["stroke_counter"] += 1
        block = sess["pending_queue"].pop(0)
        svg = _stroke_block_to_svg(block, sess["stroke_counter"], sess["grid_manager"])
        if svg:
            sess["all_strokes_svg"] += svg
            sess["stroke_history"].append({
                "stroke_no": sess["stroke_counter"],
                "block": block,
                "svg": svg
            })
            applied.append(svg)
    
    sess["pending"] = None
    
    return jsonify({
        "ok": True,
        "applied_count": len(applied),
        "all_svg": applied,
        "can_undo": len(sess["stroke_history"]) > 0
    })

@app.post("/call_model_with_context")
def call_model_with_context():
    sid, sess, _ = _ensure_session()
    if not sess.get("image_b64"):
        return jsonify({"ok": False, "error": "no image"}), 400

    adapter = _get_adapter()

    # 1) Build context text
    context_lines = [f"Already placed {sess.get('stroke_counter', 0)} strokes."]
    if sess.get("pending"):
        pb = sess["pending"]["block"] if isinstance(sess["pending"], dict) else sess["pending"]
        context_lines.append("Current pending stroke (for reference):")
        context_lines.append(str(pb))

    # ✅ Use ALL accepted strokes (model + user) via the converter
    accepted_xml_text = _accepted_strokes_text(sess)
    if accepted_xml_text.strip():
        context_lines.append("\nAccepted strokes so far (do not duplicate):")
        context_lines.append(accepted_xml_text)

    # Build user message
    user_msg = _build_user_prompt(sess["task"], sess.get("prompt", "")) + "\n\n" + "\n".join(context_lines)

    # 2) Image: try composited (grid + overlay). Fallback to base grid canvas.
    composited_b64 = _composite_canvas_with_svg(sess)
    img_b64 = composited_b64 or sess.get("canvas_with_grid_b64") or sess.get("image_b64")
    mime = sess.get("image_mime", "image/png")

    # 3) Build content
    content = [{
        "type": "image",
        "source": {"type": "base64", "media_type": mime, "data": img_b64}
    }, {"type": "text", "text": user_msg}]
    messages = [{"role": "user", "content": content}]

    # 4) Call model and prep pending
    try:
        use_stop = not isinstance(_get_adapter(), GeminiAdapter)
        additional_args = {"stop_sequences": "</answer>"} if use_stop else {}
        raw = adapter.call(_build_system_prompt(sess["grid_manager"]), messages, additional_args)
        answer = adapter.extract_text(raw) or ""
        blocks = _parse_llm_strokes(answer, sess["grid_manager"])

        sess["pending_queue"] = blocks
        next_block = blocks[0] if blocks else None

        pending_preview = ""
        if next_block:
            st = sess.get("style", {})
            svg_prev = _stroke_block_to_svg(
                next_block, 999, sess["grid_manager"],
                stroke_color=st.get("preview_color", "#f97316"),
                stroke_width=float(st.get("stroke_width", 2.0)),
                dash=bool(st.get("dash", False)),
            )
            sess["pending"] = {"block": next_block}
            sess["pending_svg_base"] = svg_prev
            sess["pending_dx"] = 0.0
            sess["pending_dy"] = 0.0
            pending_preview = svg_prev or ""
        else:
            sess["pending"] = None
            sess["pending_svg_base"] = None
            sess["pending_dx"] = 0.0
            sess["pending_dy"] = 0.0

        return jsonify({
            "ok": True,
            "queue_length": len(blocks),
            "pending": (next_block or None),
            "pending_preview": pending_preview,
            # Expose exactly what we sent so you can show it in the UI
            "accepted_xml_text": accepted_xml_text
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/delete_user_stroke")
def delete_user_stroke():
    sid, sess, _ = _ensure_session()
    data = request.get_json(silent=True) or {}
    stroke_id = str(data.get("stroke_id") or "")
    if not stroke_id:
        return jsonify({"ok": False, "error": "missing stroke_id"}), 400
    svg = sess.get("all_strokes_svg") or ""
    if not svg:
        return jsonify({"ok": False, "error": "no svg"}), 400

    import re
    pattern = rf'<g\s+[^>]*id="{re.escape(stroke_id)}"[^>]*>.*?</g>'
    new_svg, n = re.subn(pattern, '', svg, flags=re.S|re.I)
    if n == 0:
        return jsonify({"ok": False, "error": "stroke not found"}), 404

    sess["all_strokes_svg"] = new_svg
    sess["stroke_history"] = [h for h in sess.get("stroke_history", [])
                              if h.get("stroke_id") != stroke_id]

    return jsonify({"ok": True, "full_svg": new_svg, "can_undo": len(sess["stroke_history"]) > 0})



@app.post("/undo")
def undo():
    """Undo the last accepted stroke."""
    sid, sess, _ = _ensure_session()
    
    if not sess.get("stroke_history"):
        return jsonify({"ok": False, "error": "nothing to undo"}), 400
    
    # Remove last stroke from history
    last = sess["stroke_history"].pop()
    
    # Rebuild SVG without the last stroke
    sess["stroke_counter"] -= 1
    sess["all_strokes_svg"] = f'<svg width="{sess["grid_manager"].grid_size[0]}" height="{sess["grid_manager"].grid_size[1]}" xmlns="http://www.w3.org/2000/svg">'
    for item in sess["stroke_history"]:
        sess["all_strokes_svg"] += item["svg"]
    sess["all_strokes_svg"] += "</svg>"
    
    return jsonify({
        "ok": True,
        "removed_stroke": last["stroke_no"],
        "full_svg": sess["all_strokes_svg"],
        "can_undo": len(sess["stroke_history"]) > 0
    })

@app.post("/get_debug_info")
def get_debug_info():
    """Return debug information about the last API call."""
    sid, sess, _ = _ensure_session()
    
    debug = sess.get("debug_info", {})
    if not debug:
        return jsonify({"ok": False, "error": "no debug info available"}), 404
    
    return jsonify({
        "ok": True,
        "debug": debug
    })

@app.post("/download_annotated")
def download_annotated():
    """Generate downloadable PNG with all annotations."""
    sid, sess, _ = _ensure_session()
    
    if not sess.get("canvas_with_grid_b64"):
        return jsonify({"ok": False, "error": "no image"}), 400
    
    try:
        # For now, just return the SVG and current canvas
        # In production, you'd use cairosvg to composite SVG onto canvas
        return jsonify({
            "ok": True,
            "canvas_b64": sess["canvas_with_grid_b64"],
            "svg": sess.get("all_strokes_svg", ""),
            "message": "Use browser 'Save As' or screenshot to save the annotated image"
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/done")
def done():
    sid, sess, _ = _ensure_session()
    sess["accepted_strokes"] = []
    sess["pending_queue"] = []
    sess["pending"] = None
    sess["stroke_counter"] = 0
    sess["all_strokes_svg"] = ""
    sess["stroke_history"] = []
    
    # Return the canvas with grid
    data_url = f"data:image/png;base64,{sess['canvas_with_grid_b64']}" if sess.get("canvas_with_grid_b64") else None
    return jsonify({"ok": True, "data_url": data_url})

@app.get("/samples")
def get_samples():
    """
    Return grouped samples from subfolders like:
      samples/counting/*.png (+ optional .txt prompt)
      samples/labeling/*.png (+ optional .txt prompt)
      samples/custom_prompt/*.png (+ optional .txt prompt)

    Response:
      {
        "ok": true,
        "groups": [
          {"name": "Counting", "task": "counting", "items": [ {sample...}, ... ]},
          {"name": "Labeling", "task": "labeling", "items": [ {sample...}, ... ]},
          {"name": "Custom Prompt", "task": "custom", "items": [ {sample...}, ... ]}
        ]
      }

    Each sample has: { name, filename, prompt, data_url, task }
    (We also support legacy flat files in samples/ as a single "Ungrouped" group.)
    """
    from pathlib import Path

    samples_root = Path("samples")
    if not samples_root.exists():
        return jsonify({"ok": True, "groups": []})

    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

    def _load_dir(dir_path: Path, task_value: str, group_name: str):
        items = []
        if not dir_path.exists():
            return {"name": group_name, "task": task_value, "items": items}
        for img_path in sorted(dir_path.iterdir()):
            if img_path.suffix.lower() not in image_exts:
                continue
            prompt_txt = ""
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    prompt_txt = txt_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
            try:
                img_b = img_path.read_bytes()
                b64 = base64.b64encode(img_b).decode("utf-8")
                items.append({
                    "name": img_path.stem,
                    "filename": img_path.name,
                    "prompt": prompt_txt,
                    "data_url": f"data:image/{img_path.suffix[1:]};base64,{b64}",
                    "task": task_value,
                })
            except Exception as e:
                print(f"[samples] Error reading {img_path}: {e}")
        return {"name": group_name, "task": task_value, "items": items}

    # Known groups
    groups = []
    groups.append(_load_dir(samples_root / "counting", "counting", "Counting"))
    groups.append(_load_dir(samples_root / "labeling", "labeling", "Labeling"))
    groups.append(_load_dir(samples_root / "custom_prompt", "custom", "Custom Prompt"))

    # Back-compat: also scan the root for any loose images (if present)
    loose_items = []
    for img_path in sorted(samples_root.iterdir()):
        if img_path.is_dir():
            continue
        if img_path.suffix.lower() not in image_exts:
            continue
        prompt_txt = ""
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            try:
                prompt_txt = txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        try:
            b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
            loose_items.append({
                "name": img_path.stem,
                "filename": img_path.name,
                "prompt": prompt_txt,
                "data_url": f"data:image/{img_path.suffix[1:]};base64,{b64}",
                "task": "custom",  # default if ungrouped
            })
        except Exception as e:
            print(f"[samples] Error reading {img_path}: {e}")

    if loose_items:
        groups.append({"name": "Ungrouped", "task": "custom", "items": loose_items})

    # Filter out empty groups
    groups = [g for g in groups if g["items"]]

    return jsonify({"ok": True, "groups": groups})


@app.post("/add_user_stroke")
def add_user_stroke():
    """Add a user-drawn stroke to the current SVG."""
    sid, sess, _ = _ensure_session()
    data = request.json or {}
    
    st = sess.get("style", {})
    line_color = data.get("line_color") or st.get("stroke_color", "#00ff00")
    line_width = float(data.get("line_width") or st.get("stroke_width", 2.0))
    text_color = data.get("text_color") or st.get("text_color", "#0066ff")
    text_size  = int(data.get("text_size") or st.get("text_size", 16))


    if not data.get("type"):
        return jsonify({"ok": False, "error": "type required"}), 400

    gm = sess.get("grid_manager")
    if not gm:
        return jsonify({"ok": False, "error": "no grid manager"}), 400

    stroke_type = data["type"]
    stroke_svg = ""

    if stroke_type == "line" and "points" in data and len(data["points"]) >= 2:
      (x1, y1), (x2, y2) = data["points"][:2]
      # build only the inner content (no <g> here)
      stroke_svg = (
          # transparent hit halo (wide)
          f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
          f'stroke="transparent" stroke-width="{float(line_width)+12}" '
          f'vector-effect="non-scaling-stroke" pointer-events="stroke"/>'
          # visible line
          f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
          f'stroke="{line_color}" stroke-width="{line_width}" stroke-linecap="round" '
          f'vector-effect="non-scaling-stroke" marker-end="url(#arrowhead)" pointer-events="stroke"/>'
      )


    elif stroke_type == "text" and "text" in data and "point" in data:
      text = str(data["text"])
      x, y = data["point"]
      # build only the inner content (no <g> here)
      stroke_svg = (
          f'<text x="{x}" y="{y}" font-size="{text_size}" fill="{text_color}" '
          f'font-weight="bold">{text}</text>'
      )



    elif "stroke_svg" in data:
        stroke_svg = str(data["stroke_svg"])

    if not stroke_svg:
        return jsonify({"ok": False, "error": "could not generate stroke"}), 400

    # Init SVG container if needed
    if not sess.get("all_strokes_svg"):
        sess["all_strokes_svg"] = (
            f'<svg width="{gm.grid_size[0]}" height="{gm.grid_size[1]}" xmlns="http://www.w3.org/2000/svg">'
            '<defs><marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">'
            '<polygon points="0 0, 10 3, 0 6" fill="#00ff00"/></marker></defs>'
        )
    elif 'id="arrowhead"' not in sess["all_strokes_svg"]:
        # Ensure defs exists for lines with markers
        sess["all_strokes_svg"] = sess["all_strokes_svg"].replace(
            "</svg>",
            '<defs><marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">'
            '<polygon points="0 0, 10 3, 0 6" fill="#00ff00"/></marker></defs></svg>'
        )

    # Init SVG container if needed (this block already exists above)
    
    
    

    # inside add_user_stroke(), after you compute stroke_svg (line/text)
    sess["stroke_counter"] += 1
    stroke_no = sess["stroke_counter"]
    stroke_id = f"user_{stroke_no}"

    wrapped = f'<g id="{stroke_id}" class="user-stroke">{stroke_svg}</g>'

    # open container if needed; then append
    svg = sess.get("all_strokes_svg") or f'<svg width="{gm.grid_size[0]}" height="{gm.grid_size[1]}" xmlns="http://www.w3.org/2000/svg">'
    if svg.endswith("</svg>"):
        svg = svg[:-6]
    sess["all_strokes_svg"] = svg + wrapped + "</svg>"

    sess.setdefault("stroke_history", []).append({
        "stroke_no": stroke_no,
        "stroke_id": stroke_id,
        "svg": wrapped,
        "type": stroke_type,
        "source": "user"
    })

    return jsonify({"ok": True, "stroke_no": stroke_no, "stroke_id": stroke_id,
                    "svg": wrapped, "full_svg": sess["all_strokes_svg"], "can_undo": True})


        


@app.post("/add_freehand")
def add_freehand():
    sid, sess, _ = _ensure_session()
    data = request.get_json(silent=True) or {}
    path = data.get("path") or []
    if not path:
        return jsonify({"ok": False, "error": "missing path"}), 400

    gm = sess.get("grid_manager")
    if gm is None:
        return jsonify({"ok": False, "error": "no grid manager"}), 400

    # Use the UI-provided style if present; otherwise fall back to session style.
    st = sess.get("style", {})
    stroke_color = data.get("line_color") or st.get("stroke_color", "#00aa00")
    stroke_width = float(data.get("line_width") or st.get("stroke_width", 2.0))

    # Build the exact visual polyline (pixel coords) — keep user-picked color.
    pts_attr = " ".join(
        f"{p['x']:.1f},{p['y']:.1f}" if isinstance(p, dict) else f"{p[0]:.1f},{p[1]:.1f}"
        for p in path
    )

    next_no = int(sess.get("stroke_counter", 0)) + 1
    stroke_id = f"user_{next_no}"

    # Build exact visual polyline + a wide transparent "halo" for easy selection
    halo_width = max(14.0, float(stroke_width) + 10.0)

    raw_polyline = (
        f'<g id="{stroke_id}" class="user-stroke freehand" data-source="user_freehand">'
        # selection halo
        f'<polyline points="{pts_attr}" fill="none" '
        f'stroke="transparent" stroke-width="{halo_width}" pointer-events="stroke"/>'
        # visible stroke
        f'<polyline points="{pts_attr}" fill="none" '
        f'stroke="{stroke_color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'</g>'
    )

    # Text form for the model (dedup consecutive tokens)
    tokens, _tvals = _px_to_grid_tokens(path, gm)
    tokens, tvals = _dedupe_consecutive_tokens_and_t(tokens, _tvals)
    xml_block = (
        f"<s{next_no}>\n"
        f"  <points>{','.join(tokens)}</points>\n"
        f"  <t_values>{','.join(tvals)}</t_values>\n"
        f"</s{next_no}>"
    )

    # Ensure master SVG exists, then append
    if not sess.get("all_strokes_svg"):
        sess["all_strokes_svg"] = (
            f'<svg width="{gm.grid_size[0]}" height="{gm.grid_size[1]}" '
            f'xmlns="http://www.w3.org/2000/svg"></svg>'
        )

    svg = sess["all_strokes_svg"]
    if svg.endswith("</svg>"):
        svg = svg[:-6]
    svg += raw_polyline + "</svg>"
    sess["all_strokes_svg"] = svg

    # Record and return
    sess["stroke_counter"] = next_no
    sess.setdefault("stroke_history", []).append({
        "stroke_no": next_no,
        "stroke_id": stroke_id,
        "block": xml_block,          # CLEAN text (deduped)
        "svg": raw_polyline,         # visual (as drawn, with color)
        "source": "user_freehand"
    })
    sess.setdefault("user_xml_strokes", []).append(xml_block)

    return jsonify({
        "ok": True,
        "xml_block": xml_block,
        "full_svg": sess["all_strokes_svg"],
        "stroke_id": stroke_id,
        "can_undo": len(sess["stroke_history"]) > 0
    })





@app.post("/update_pending_position")
def update_pending_position():
    sid, sess, _ = _ensure_session()
    data = request.json or {}
    if "delta_x" not in data or "delta_y" not in data:
        return jsonify({"ok": False, "error": "delta_x and delta_y required"}), 400
    if not sess.get("pending"):
        return jsonify({"ok": False, "error": "no pending stroke"}), 400
    if not sess.get("pending_svg_base"):
        return jsonify({"ok": False, "error": "no pending preview available"}), 400

    # Accumulate deltas
    sess["pending_dx"] = float(sess.get("pending_dx", 0.0) or 0.0) + float(data["delta_x"])
    sess["pending_dy"] = float(sess.get("pending_dy", 0.0) or 0.0) + float(data["delta_y"])

    # Bake translation into base SVG and reset deltas so it "sticks"
    updated = _compose_pending_svg(sess)
    sess["pending_svg_base"] = updated
    sess["pending_dx"] = 0.0
    sess["pending_dy"] = 0.0

    return jsonify({"ok": True, "updated_svg": updated})
  
  
@app.post("/transform_user_stroke")
def transform_user_stroke():
    sid, sess, _ = _ensure_session()
    data = request.get_json(silent=True) or {}
    stroke_id = str(data.get("id") or "")
    dx = float(data.get("dx") or 0.0)
    dy = float(data.get("dy") or 0.0)
    if not stroke_id:
        return jsonify({"ok": False, "error": "missing id"}), 400
    svg = sess.get("all_strokes_svg") or ""
    if not svg:
        return jsonify({"ok": False, "error": "no svg"}), 400

    # find the <g id="stroke_id">...</g> and add/compose a translate()
    import re
    def apply_transform(gmatch):
        gtag = gmatch.group(0)
        # Pull current transform attribute (if any)
        m_attr = re.search(r'\btransform="([^"]*)"', gtag)
        if not m_attr:
            # no transform yet → just set the delta
            return gtag.replace('<g ', f'<g transform="translate({dx},{dy})" ')

        full = m_attr.group(1)
        # Sum all existing translate()s
        tx_total = 0.0
        ty_total = 0.0
        for tx, ty in re.findall(r'translate\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', full):
            tx_total += float(tx)
            ty_total += float(ty)
        tx_total += dx
        ty_total += dy

        # Remove all existing translate() pieces
        rest = re.sub(r'translate\(\s*[-\d.]+\s*,\s*[-\d.]+\s*\)\s*', '', full).strip()
        # Rebuild with a single combined translate first, then any remaining transforms
        new_full = f'translate({tx_total},{ty_total})'
        if rest:
            new_full += ' ' + rest
        return gtag.replace(m_attr.group(0), f'transform="{new_full}"')


    pattern = rf'<g\s+[^>]*id="{re.escape(stroke_id)}"[^>]*>.*?</g>'
    new_svg, n = re.subn(pattern, apply_transform, svg, flags=re.S|re.I)
    if n == 0:
        return jsonify({"ok": False, "error": "stroke not found"}), 404

    sess["all_strokes_svg"] = new_svg

    # update history copy for that stroke
    for it in sess.get("stroke_history", []):
        if it.get("stroke_id") == stroke_id:
            # regenerate its stored svg by re-extracting the group
            gm = re.search(pattern, new_svg, flags=re.S|re.I)
            if gm:
                it["svg"] = gm.group(0)
            break

    return jsonify({"ok": True, "full_svg": new_svg})



# @app.post("/delete_user_stroke")
# def delete_user_stroke():
#     sid, sess, _ = _ensure_session()
#     data = request.json or {}
#     stroke_id = str(data.get("id", ""))
#     if not stroke_id:
#         return jsonify({"ok": False, "error": "id required"}), 400
#     svg = sess.get("all_strokes_svg", "")
#     if not svg:
#         return jsonify({"ok": False, "error": "no svg"}), 400

#     pattern = rf'<g[^>]*\bid="{re.escape(stroke_id)}"[^>]*>.*?</g>'
#     new_svg, n = re.subn(pattern, '', svg, count=1, flags=re.S)
#     if n == 0:
#         return jsonify({"ok": False, "error": "stroke not found"}), 404

#     sess["stroke_history"] = [h for h in sess.get("stroke_history", []) if h.get("id") != stroke_id]
#     sess["all_strokes_svg"] = new_svg
#     return jsonify({"ok": True, "full_svg": new_svg, "can_undo": len(sess.get("stroke_history", [])) > 0})

@app.post("/delete_pending")
def delete_pending():
    sid, sess, _ = _ensure_session()
    if not sess.get("pending"):
        return jsonify({"ok": False, "error": "no pending stroke"}), 400
    # drop current from queue if present
    if sess.get("pending_queue"):
        try:
            cur = sess["pending"]["block"] if isinstance(sess["pending"], dict) else sess["pending"]
            if sess["pending_queue"] and sess["pending_queue"][0] == cur:
                sess["pending_queue"].pop(0)
        except Exception:
            sess["pending_queue"].pop(0)
    next_block = sess["pending_queue"][0] if sess.get("pending_queue") else None

    # reset pending state
    sess["pending"] = None
    sess["pending_svg_base"] = None
    sess["pending_dx"] = 0.0
    sess["pending_dy"] = 0.0

    pending_preview = None
    if next_block:
        st = sess.get("style", {})
        gm = sess["grid_manager"]
        pending_preview = _stroke_block_to_svg(
            next_block, 999, gm,
            stroke_color=st.get("preview_color", "#f97316"),
            stroke_width=float(st.get("stroke_width", 2.0)),
            dash=bool(st.get("dash", False)),
        )
        sess["pending"] = {"block": next_block}
        sess["pending_svg_base"] = pending_preview

    return jsonify({
        "ok": True,
        "pending": next_block,
        "pending_preview": pending_preview,
        "queue_remaining": len(sess.get("pending_queue", [])),
    })




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sketch Grid Demo with Multi/One-turn modes")
    parser.add_argument("--llm", choices=["claude", "gpt", "gemini"], default="gemini",
                        help="LLM provider (default: gemini)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (default: provider-specific default)")
    parser.add_argument("--cell-size", type=int, default=15,
                        help="Grid cell size in pixels (default: 15)")
    parser.add_argument("--res", type=int, default=50,
                        help="Grid resolution (default: 50)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max tokens for model response (default: 8192)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Server port (default: 5000)")
    parser.add_argument("--show-full-grid", action="store_true",
                        help="Show all grid lines (default: axis-only style)")
    parser.add_argument("--adaptive-grid", action="store_true", default=True,
                        help="Auto-scale grid cell size based on image resolution (default: True)")
    parser.add_argument("--target-cols", type=int, default=50,
                        help="Desired max columns when adaptive-grid is on (default: 50)")
    parser.add_argument("--target-rows", type=int, default=50,
                        help="Desired max rows when adaptive-grid is on (default: 50)")
    parser.add_argument("--min-cell-px", type=int, default=20,
                        help="Minimum pixel size per cell for readability (default: 20)")
    parser.add_argument("--max-cell-px", type=int, default=64,
                        help="Upper bound for pixel size per cell (default: 64)")
    
    args = parser.parse_args()
    
    # Set model defaults per provider if not specified
    if args.model is None:
        if args.llm == "claude":
            args.model = "claude-3-5-sonnet-20240620"
        elif args.llm == "gpt":
            args.model = "gpt-4o"
        elif args.llm == "gemini":
            args.model = "gemini-2.5-pro"
    
    # Update global config
    CONFIG.update({
        "llm": args.llm,
        "model": args.model,
        "cell_size": args.cell_size,
        "res": args.res,
        "max_tokens": args.max_tokens,
        "show_full_grid": args.show_full_grid,
        "adaptive_grid": args.adaptive_grid,
        "target_cols": args.target_cols,
        "target_rows": args.target_rows,
        "min_cell_px": args.min_cell_px,
        "max_cell_px": args.max_cell_px,
    })
    
    print(f"\n{'='*60}")
    print(f"  Sketch Grid Demo - Interactive Annotation Tool")
    print(f"{'='*60}")
    print(f"  Provider:     {CONFIG['llm']}")
    print(f"  Model:        {CONFIG['model']}")
    print(f"  Grid Style:   {'Full grid' if CONFIG['show_full_grid'] else 'Axis-only (recommended)'}")
    print(f"  Adaptive:     {CONFIG['adaptive_grid']}")
    if CONFIG['adaptive_grid']:
        print(f"  Target Grid:  {CONFIG['target_cols']}×{CONFIG['target_rows']} cells")
        print(f"  Cell Range:   {CONFIG['min_cell_px']}-{CONFIG['max_cell_px']}px")
    else:
        print(f"  Static Grid:  {CONFIG['res']}×{CONFIG['res']} cells @ {CONFIG['cell_size']}px")
    print(f"  Port:         {args.port}")
    print(f"{'='*60}")
    print(f"\n⚠️  Make sure you have the appropriate API key set:")
    if args.llm == "gemini":
        print("     GOOGLE_API_KEY or GEMINI_API_KEY")
    elif args.llm == "claude":
        print("     ANTHROPIC_API_KEY")
    elif args.llm == "gpt":
        print("     OPENAI_API_KEY")
    print(f"\n🌐 Open your browser to: http://localhost:{args.port}")
    print(f"{'='*60}\n")
    
    app.run(host="0.0.0.0", port=args.port, debug=True)
    
    
    
    
