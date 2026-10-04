const $ = id => document.getElementById(id);
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
let latest = {}, run = null, previewURL = null;
let recording = {}, recordingBusy = false, lastTarget = null, recordMessage = '';
let previewEnabled = true, cameraBusy = false, settingsLoaded = false, selectROI = false, selectPupil = false, roiStart = null, roi = [.1, .25, .9, .75];
async function api(path, data) {
 const response = await fetch(path, data === undefined ? {} : {
  method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)
 });
 if (!response.ok) throw Error(await response.text());
 return response.json();
}
const cal = data => api('/api/calibration', data);
function updateFeedButton() {
 const connected = latest.camera_connected && typeof latest.frame_age_ms === 'number' && latest.frame_age_ms < 350;
 $('camera-feed').textContent = cameraBusy ? 'USB 카메라를 다시 찾는 중…' : connected ? (previewEnabled ? '영상 피드 끄기' : '영상 피드 켜기') : '카메라 연결 새로고침';
 $('camera-feed').disabled = cameraBusy || Boolean(run);
 $('camera-feed').setAttribute('aria-pressed', String(Boolean(connected && previewEnabled)));
 $('preview-wrap').hidden = !connected || !previewEnabled;
}
$('camera-feed').onclick = async () => {
 if (latest.camera_connected && typeof latest.frame_age_ms === 'number' && latest.frame_age_ms < 350) {
  previewEnabled = !previewEnabled;
  for (const id of ['preview', 'calpreview']) $(id).hidden = !previewEnabled;
  $('roi-box').hidden = !previewEnabled || roi.join() === '0,0,1,1';
  updateFeedButton(); return;
 }
 cameraBusy = true; updateFeedButton();
 try { await api('/api/camera/refresh', {}); previewEnabled = true; $('setup-message').textContent = 'USB 재검색 요청을 보냈습니다. 아래 연결 상태를 확인하세요.'; }
 catch (error) { $('setup-message').textContent = error.message; }
 finally { cameraBusy = false; updateFeedButton(); }
};
function drawROI() {
 $('roi-box').hidden = !previewEnabled || roi.join() === '0,0,1,1';
 Object.assign($('roi-box').style, {left: `${roi[0]*100}%`, top: `${roi[1]*100}%`, width: `${(roi[2]-roi[0])*100}%`, height: `${(roi[3]-roi[1])*100}%`});
 for (const [index, id] of ['roi-left', 'roi-top', 'roi-right', 'roi-bottom'].entries()) $(id).value = (roi[index]*100).toFixed(1);
}
for (const id of ['roi-left', 'roi-top', 'roi-right', 'roi-bottom']) $(id).onchange = () => {
 roi = ['roi-left', 'roi-top', 'roi-right', 'roi-bottom'].map(key => +$(key).value / 100); drawROI();
};
$('select-roi').onclick = () => {
 selectPupil = false;
 selectROI = true; $('preview-wrap').classList.add('selecting'); previewEnabled = true;
 $('setup-message').textContent = '영상에서 눈 영역을 드래그한 뒤 설정 적용을 누르세요.'; updateFeedButton();
};
$('select-pupil').onclick = () => {
 selectROI = false; selectPupil = true; previewEnabled = true;
 $('preview-wrap').classList.add('selecting');
 $('setup-message').textContent = '작은 검은 동공 전체만 사각형으로 감싸세요. 홍채나 반사점을 고르면 잘못된 대상을 추적합니다.';
 updateFeedButton();
};
$('central-roi').onclick = () => { roi = [.25, .25, .75, .75]; drawROI(); };
$('wide-roi').onclick = () => { roi = [.1, .25, .9, .75]; drawROI(); };
$('auto-intensity').onchange = () => { $('intensity-range').disabled = $('auto-intensity').checked && !$('auto-intensity').disabled; };
$('engine').onchange = () => {
 const experimental = ['orlosky-stable', 'orlosky-stable-3d', 'orlosky-flow', 'orlosky-ecc', 'orlosky-tapir', 'deepvog', 'deepvog-ecc', 'deepvog-verified', 'ritnet'].includes($('engine').value);
 if (experimental && roi.join() === '0,0,1,1') { roi = [.1, .25, .9, .75]; drawROI(); }
 $('auto-intensity').disabled = !['orlosky-stable', 'orlosky-stable-3d', 'orlosky-flow', 'orlosky-ecc', 'orlosky-tapir'].includes($('engine').value);
 $('auto-intensity').onchange();
 $('compensate-motion').disabled = $('engine').value !== 'orlosky-flow';
 if (['orlosky-flow', 'orlosky-ecc', 'orlosky-tapir'].includes($('engine').value)) { $('pupil-min').value = 25; $('pupil-max').value = 100; }
 if ($('engine').value.startsWith('deepvog') && $('engine').value !== 'deepvog-verified') { $('pupil-min').value = 10; $('pupil-max').value = 160; }
 if ($('engine').value === 'deepvog-verified') { $('pupil-min').value = 10; $('pupil-max').value = 320; }
 $('focal-length').disabled = !$('engine').value.endsWith('-3d');
 $('setup-message').textContent = experimental ? '비교 경로입니다. 머리카락·안경테가 제외되고 모든 주시 방향의 동공이 포함되도록 눈 영역을 확인하세요. 경로 변경 후 1→9→4를 새로 진행하세요.' : '';
};
$('clear-roi').onclick = () => { roi = [0, 0, 1, 1]; roiStart = null; selectROI = false; $('preview-wrap').classList.remove('selecting'); drawROI(); };
function imagePoint(event) {
 const box = $('preview-wrap').getBoundingClientRect();
 return [Math.max(0, Math.min(1, (event.clientX-box.left)/box.width)), Math.max(0, Math.min(1, (event.clientY-box.top)/box.height))];
}
$('preview-wrap').onpointerdown = event => {
 if ((!selectROI && !selectPupil) || !latest.camera_connected || !previewEnabled) return;
 roiStart = imagePoint(event); $('preview-wrap').setPointerCapture(event.pointerId);
};
$('preview-wrap').onpointermove = event => {
 if (!roiStart) return;
 const end = imagePoint(event);
 const box = [Math.min(roiStart[0],end[0]), Math.min(roiStart[1],end[1]), Math.max(roiStart[0],end[0]), Math.max(roiStart[1],end[1])];
 if (selectPupil) Object.assign($('roi-box').style, {left: `${box[0]*100}%`, top: `${box[1]*100}%`, width: `${(box[2]-box[0])*100}%`, height: `${(box[3]-box[1])*100}%`});
 else { roi = box; drawROI(); }
};
$('preview-wrap').onpointerup = async event => {
 if (roiStart && selectPupil) {
  const end = imagePoint(event), rectangle = [Math.min(roiStart[0],end[0]), Math.min(roiStart[1],end[1]), Math.max(roiStart[0],end[0]), Math.max(roiStart[1],end[1])];
  try { await api('/api/tracking/seed', {rectangle}); $('setup-message').textContent = '선택한 동공으로 연속 추적을 시작했습니다. 타원을 확인하고 1→9→4를 진행하세요.'; }
  catch (error) { $('setup-message').textContent = error.message; }
 }
 roiStart = null; selectROI = selectPupil = false; $('preview-wrap').classList.remove('selecting'); drawROI();
};
$('preview-wrap').onpointercancel = () => { roiStart = null; };
$('apply-detector').onclick = async () => {
 $('apply-detector').disabled = true;
 try {
  await api('/api/detector', {engine: $('engine').value, pupil_min: +$('pupil-min').value, pupil_max: +$('pupil-max').value,
   intensity_range: +$('intensity-range').value, auto_intensity: !$('auto-intensity').disabled && $('auto-intensity').checked,
   compensate_motion: !$('compensate-motion').disabled && $('compensate-motion').checked,
   focal_length: +$('focal-length').value, roi});
  $('setup-message').textContent = '설정을 적용했습니다. 타원이 작은 동공을 따라가는지 확인한 뒤 새로 보정하세요.';
  $('cursor').hidden = true;
 } catch (error) { $('setup-message').textContent = error.message; }
 finally { $('apply-detector').disabled = false; }
};
function refreshRecording(state = recording) {
 recording = state;
 for (const button of document.querySelectorAll('[data-record]')) {
  const action = button.dataset.record, active = action.endsWith('_all') ? (action.startsWith('start_') ? state.log_active && state.video_active : state.log_active || state.video_active) : action.endsWith('_log') ? state.log_active : state.video_active;
  button.disabled = recordingBusy || (action.startsWith('start_') ? Boolean(active) : !active);
 }
 for (const link of document.querySelectorAll('.record-download')) {
  link.hidden = !state.download_url;
  if (state.download_url) { link.href = state.download_url; link.download = `${state.recording_id}.zip`; }
 }
 for (const element of document.querySelectorAll('.record-status')) {
  element.textContent = state.error || recordMessage || `${state.log_active ? '● 로그 기록 중' : '로그 정지'} · ${state.video_active ? `● 영상 녹화 중 (${state.video_frames}프레임)` : '영상 정지'}${state.recording_id ? `\n${state.recording_id}` : ''}`;
  element.classList.toggle('active', Boolean(state.log_active || state.video_active));
  element.classList.toggle('error', Boolean(state.error || recordMessage));
 }
}
function targetEvent(phase, point, message, index = null) {
 const event = {phase, index, point: point || null, message, client_timestamp_ms: Date.now(), viewport: {width: innerWidth, height: innerHeight}};
 const key = JSON.stringify([phase, index, event.point, message]);
 if (lastTarget?.key === key) return;
 lastTarget = {key, event};
 if (recording.log_active) api('/api/recording/event', event).catch(error => { recordMessage = `화면 단계 기록 실패: ${error.message}`; refreshRecording(); });
}
for (const button of document.querySelectorAll('[data-record]')) button.onclick = async () => {
 if (recordingBusy) return;
 recordingBusy = true; recordMessage = ''; refreshRecording();
 try {
  const state = await api('/api/recording', {action: button.dataset.record});
  refreshRecording(state);
  if (['start_log', 'start_all'].includes(button.dataset.record)) {
   const event = lastTarget?.event || {phase: run?.phase || 'fit', index: null, point: null, message: '기록 시작', viewport: {width: innerWidth, height: innerHeight}};
   await api('/api/recording/event', {...event, client_timestamp_ms: Date.now()});
  }
 } catch (error) { recordMessage = error.message; }
 finally { recordingBusy = false; refreshRecording(); }
};
function show(message, point, retry = false, confirm = false) {
 $('message').textContent = message;
 $('dot').hidden = !point;
 if (point) { $('dot').style.left = point[0] * 100 + '%'; $('dot').style.top = point[1] * 100 + '%'; }
 $('panel').style.top = point && point[1] > .6 ? '3%' : 'auto';
 $('panel').style.bottom = point && point[1] > .6 ? 'auto' : '3%';
 $('retry').hidden = !retry;
 $('continue').hidden = !confirm;
 (retry ? $('retry') : confirm ? $('continue') : $('cancel')).focus();
 targetEvent(run?.phase || 'fit', point, message, run?.index ?? null);
}
function waitAction(current, kind) {
 return new Promise((resolve, reject) => { current.waiter = {kind, resolve, reject}; });
}
for (const [id, kind] of [['continue', 'continue'], ['retry', 'retry']]) $(id).onclick = () => {
 if (run?.waiter?.kind === kind) { run.waiter.resolve(); run.waiter = null; }
};
async function cancel(message = '취소했습니다. 진단 JSON에서 이전 시도를 확인할 수 있습니다.') {
 const current = run;
 if (!current) return;
 run = null;
 if (current.phase === 'drift') { $('calibration').hidden = true; $('result').textContent = '드리프트 보정을 취소했습니다. 기존 보정은 유지됩니다.'; return; }
 targetEvent('cancel', null, message);
 current.waiter?.reject(Error('cancelled'));
 $('calibration').hidden = true;
 $('start').disabled = true;
 $('result').textContent = message;
 try { await cal({action: 'cancel', ...(current.session ? {session_id: current.session} : {})}); }
 catch (error) { $('result').textContent = error.message; }
 finally { $('start').disabled = $('fullscreen').disabled = false; $('start').focus(); }
}
$('cancel').onclick = () => cancel();
addEventListener('keydown', event => {
 if (!run) return;
 if (event.key === 'Escape') cancel();
 if (event.key === 'Tab') {
  const buttons = [...$('calibration').querySelectorAll('button, a[href]')].filter(button => !button.hidden && !button.disabled);
  const first = buttons[0], last = buttons.at(-1);
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
 }
});
$('fullscreen').onclick = async () => {
 try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); }
 catch (error) { $('result').textContent = error.message; }
};
addEventListener('resize', () => {
 if (run) cancel('화면 크기가 바뀌었습니다. 같은 화면에서 다시 보정하세요.');
 else if (latest.calibrated) {
  $('cursor').hidden = true;
  cal({action: 'cancel', session_id: latest.session_id}).catch(error => $('result').textContent = error.message);
  $('result').textContent = '화면 크기가 바뀌었습니다. 다시 보정하세요.';
 }
});
$('start').onclick = async () => {
 if (run || $('start').disabled) return;
 if (latest.camera?.mode === 'video') { $('result').textContent = '영상 파일 재생 모드입니다. USB 카메라 입력에서 보정하세요.'; return; }
 $('start').disabled = $('fullscreen').disabled = true;
 const current = {session: null, phase: 'fit', index: null, viewport: {width: innerWidth, height: innerHeight}};
 run = current;
 const check = () => { if (run !== current) throw Error('cancelled'); };
 const attempt = async (task, message, point) => {
  while (true) {
   check();
   try { const result = await task(); check(); return result; }
   catch (error) { check(); show(`${message}\n${error.message}`, point, true); await waitAction(current, 'retry'); }
  }
 };
 $('calibration').hidden = false;
 $('cursor').hidden = true;
 try {
  await cal({action: 'reset'}); check();
  const plan = await api('/api/plan'); check();
  show('눈 영상에서 타원이 동공을 따라가는지 확인하세요.', null, false, true);
  await waitAction(current, 'continue'); check();
  const warm = [[.5, .5], [.2, .5], [.8, .5], [.5, .2], [.5, .8]];
  let step = 0;
  current.phase = 'model';
  while (!latest.ready) {
   show(`눈 모델 준비 · 머리는 고정하고 점을 따라 눈을 움직이세요.\n${latest.error || ''}`, warm[Math.floor(step / 18) % warm.length]);
   await pause(100); check(); step++;
  }
  const session = await attempt(() => cal({action: 'begin', viewport: current.viewport}), '눈 모델 고정', [.5, .5]);
  current.session = session.session_id;
  current.phase = 'neutral';
  const neutralMessage = '1단계 · 중앙 1점\n머리를 유지하고 중앙을 바라보세요. 착용 기준을 수집합니다.';
  show(neutralMessage, [.5, .5]);
  await attempt(async () => { await pause(400); check(); return cal({action: 'neutral', session_id: current.session}); }, neutralMessage, [.5, .5]);
  for (let i = 0; i < plan.points.length; i++) {
   current.phase = 'sample'; current.index = i;
   const point = plan.points[i], message = `2단계 · 3×3 보정 ${i + 1} / 9\n점을 계속 바라보세요.`;
   show(message, point);
   await attempt(async () => { await pause(900); check(); return cal({action: 'sample', index: i, session_id: current.session}); }, message, point);
   show('수집했습니다.', point); await pause(200); check();
  }
  let result;
  for (let i = 0; i < plan.validation_points.length; i++) {
   current.phase = 'validate'; current.index = i;
   const point = plan.validation_points[i], message = `3단계 · 대각선 검증 ${i + 1} / 4 (기준: ${plan.tolerance})\n점을 계속 바라보세요. 실패가 반복되면 취소하고 9점부터 다시 보정하세요.`;
   show(message, point);
   result = await attempt(async () => { await pause(900); check(); return cal({action: 'validate', validation_index: i, collect_all: true, session_id: current.session}); }, message, point);
  }
  if (!result?.calibrated) {
   check(); run = null; $('calibration').hidden = true; $('cursor').hidden = true;
   const errors = result?.validation_errors || [];
   const passes = result?.validation_passes || [];
   const message = `4점 검증 기록 완료 · 보정 실패 (기준: ${result?.tolerance || '거리 0.120'})\n${errors.map((value, i) => `${i+1}번: 거리 ${value.toFixed(3)} · ${passes[i] ? '통과' : '실패'}`).join('\n')}\n검출 후보와 카메라 위치를 확인한 뒤 새로 보정하세요.`;
   $('result').textContent = message; targetEvent('complete', null, message);
   $('start').disabled = $('fullscreen').disabled = false; return;
  }
  check(); run = null;
  targetEvent('complete', null, '보정 완료');
  $('calibration').hidden = true;
  $('result').textContent = `완료 · 대각선 최대 정규 좌표 오차 ${result.validation_error.toFixed(3)} · 기준 ${result.tolerance}${result.excluded_point != null ? `\n9점 중 ${result.excluded_point + 1}번 점은 다른 점과 맞지 않아 제외했습니다.` : ''}\n노란 커서를 눈으로 움직여보세요. 시간이 지나 어긋나면 '드리프트 보정'을 누르세요.`;
  $('start').disabled = $('fullscreen').disabled = false; $('start').focus();
 } catch (error) { if (run === current) await cancel(error.message); }
};
let regionGrid = null;
function drawRegions(grid) {
 const key = grid?.join('x');
 if (!grid || key === regionGrid) return;
 regionGrid = key;
 $('regions').style.gridTemplateColumns = `repeat(${grid[0]}, 1fr)`;
 $('regions').style.gridTemplateRows = `repeat(${grid[1]}, 1fr)`;
 $('regions').replaceChildren(...Array.from({length: grid[0] * grid[1]}, () => document.createElement('div')));
}
$('drift').onclick = async () => {
 if (run || !latest.calibrated) return;
 const current = {phase: 'drift', index: null, session: null};
 run = current;
 $('calibration').hidden = false; $('cursor').hidden = true;
 const point = [.5, .5];
 try {
  show('드리프트 보정 · 중앙 점을 바라보세요.', point);
  await pause(900);
  if (run !== current) return;
  const result = await cal({action: 'drift', target: point});
  $('result').textContent = `드리프트 보정 완료 · 화면 오프셋 가로 ${result.offset[0].toFixed(3)}, 세로 ${result.offset[1].toFixed(3)}`;
 } catch (error) { $('result').textContent = error.message; }
 finally { if (run === current) { run = null; $('calibration').hidden = true; } }
};
async function poll() {
 try {
  latest = await api('/api/status');
  updateFeedButton();
  const replay = latest.camera?.mode === 'video';
  $('start').disabled = Boolean(run) || replay;
  const camera = latest.camera || {};
  $('source-status').textContent = replay ? `영상 파일 재생: ${camera.source} · 녹화는 이 파일을 다시 저장합니다. 실제 USB 피드가 아닙니다.` :
   `USB 카메라 입력: ${camera.device?.name || '연결 대기'}${camera.capture_width ? ` · ${camera.capture_width}×${camera.capture_height}` : ''}${latest.camera_connected ? '' : ' · 현재 연결 안 됨'}`;
  for (const button of document.querySelectorAll('[data-record="start_video"]')) button.textContent = replay ? '재생 영상 저장 시작' : '영상 녹화 시작';
  if (!settingsLoaded && latest.detector) {
   const settings = latest.detector.settings;
   $('engine').value = latest.detector.engine;
   $('focal-length').value = latest.detector.focal_length;
   $('focal-length').disabled = !latest.detector.engine.endsWith('-3d');
   $('auto-intensity').checked = Boolean(settings.auto_intensity);
   $('compensate-motion').checked = Boolean(settings.compensate_motion);
   $('compensate-motion').disabled = latest.detector.engine !== 'orlosky-flow';
   $('auto-intensity').disabled = !['orlosky-stable', 'orlosky-stable-3d', 'orlosky-flow', 'orlosky-ecc', 'orlosky-tapir'].includes(latest.detector.engine);
   $('auto-intensity').onchange();
   if (settings.pupil_min != null) { $('pupil-min').value = settings.pupil_min; $('pupil-max').value = settings.pupil_max; $('intensity-range').value = settings.intensity_range; roi = settings.roi; drawROI(); }
   settingsLoaded = true;
  }
  refreshRecording(latest.recording || {});
  $('select-pupil').disabled = !['orlosky-flow', 'orlosky-ecc', 'orlosky-tapir'].includes(latest.detector?.engine) || !latest.camera_connected || Boolean(run);
  $('focus-status').textContent = latest.camera?.autofocus_request_accepted ? '렌즈 자동 초점 요청이 접수되었습니다. 3D 모델 초점거리는 별도로 측정해야 합니다.' : '현재 입력 경로는 렌즈 자동 초점 제어를 지원하지 않습니다. 렌즈·거리를 직접 맞추세요. 3D 모델 초점거리는 렌즈 조절값이 아닙니다.';
  const age = latest.frame_age_ms;
  const temporal = latest.tracker_details || {};
  const flowStatus = temporal.temporal_source ? `\n연속 추적: ${temporal.temporal_source}${temporal.reference_required ? '' : ` · 피부 이동 ${temporal.skin_motion_px ? temporal.skin_motion_px.map(v => v.toFixed(1)).join(', ')+' px' : '추정 불가'}${temporal.motion_compensation_enabled ? ' · 흔들림 보정 켜짐' : ''}`}` : '';
  const referenceStatus = temporal.reference_required ? `\n착용 기준: ${temporal.reference_ready ? '저장됨' : '중앙 1점에서 수집'} · 주변 이동 ${temporal.reference_motion_sensor_px ? temporal.reference_motion_sensor_px.map(v => v.toFixed(1)).join(', ')+' px' : '대기/확인 중'}` : '';
  const input = latest.camera?.mode === 'video' ? '녹화 영상 재생 · 실제 시선 측정 아님' : latest.camera_connected ? '카메라 연결됨' : '카메라 대기';
  const pupil = latest.pupil, model = latest.input_kind === 'pupil_center_2d' ? '2D 동공 좌표' : '3D 눈 모델';
  $('status').textContent = `${input} · ${latest.detector?.engine || ''} · ${latest.processing_fps} FPS · ${latest.processing_ms ?? '—'} ms (검출+미리보기)\n검출 품질 ${(latest.detection_quality ?? 0).toFixed(2)} · ${model} ${latest.ready ? '준비 완료' : '준비 중'}${pupil ? ` · 동공 지름 ${Math.max(...pupil.axes).toFixed(1)} px` : ''}\n${latest.error || ''}${flowStatus}${referenceStatus}`;
  if (latest.fit_diagnostics) $('status').textContent += `\n9점 학습 오차 ${latest.fit_diagnostics.fit_error.toFixed(3)} · 한 점 생략 검사 ${latest.fit_diagnostics.loo_max_error?.toFixed(3) ?? '—'} (독립 검증과 별개)`;
  const attempt = latest.last_attempt;
  if (attempt) $('status').textContent += `\n최근 ${attempt.action === 'validate' ? '검증' : '보정'} ${(attempt.index ?? 0) + 1}: 유효 ${attempt.valid_frames}/${attempt.total_frames} 프레임 · 흔들림 ${attempt.raw_p90?.toFixed(4) ?? '—'}${attempt.error_norm == null ? '' : ` · 오차 ${attempt.error_norm.toFixed(3)} / ${attempt.error_px?.toFixed(1) ?? '—'} CSS px`}`;
  if (run?.session && latest.session_id !== run.session) await cancel('카메라 또는 보정 세션이 초기화되었습니다. 다시 시작하세요.');
  const viewport = latest.calibration_viewport;
  const matches = !viewport || (viewport.width === innerWidth && viewport.height === innerHeight);
  $('cursor').hidden = Boolean(run) || !latest.valid || !matches || age > 350;
  $('drift').hidden = !latest.calibrated;
  drawRegions(latest.grid);
  $('regions').hidden = !$('show-regions').checked || Boolean(run) || !latest.calibrated;
  [...$('regions').children].forEach((cell, i) => cell.classList.toggle('active',
   Boolean(latest.region) && !$('cursor').hidden && i === latest.region[1] * latest.grid[0] + latest.region[0]));
  if (!$('cursor').hidden) { $('cursor').style.left = latest.x * innerWidth + 'px'; $('cursor').style.top = latest.y * innerHeight + 'px'; }
 } catch (error) {
  latest = {};
  updateFeedButton();
  $('cursor').hidden = true;
  $('status').textContent = `서버 연결을 확인하세요. ${error.message}`;
 } finally { setTimeout(poll, 50); }
}
async function preview() {
 try {
  if (!document.hidden && latest.camera_connected && previewEnabled) {
   const response = await fetch('/preview.jpg');
   if (!response.ok) throw Error();
   const url = URL.createObjectURL(await response.blob()), previous = previewURL;
   previewURL = url;
   for (const id of ['preview', 'calpreview']) { $(id).src = url; $(id).hidden = false; }
   if (previous) URL.revokeObjectURL(previous);
  } else if (!latest.camera_connected || !previewEnabled) { $('roi-box').hidden = true; throw Error(); }
 } catch { $('preview').hidden = $('calpreview').hidden = true; }
 finally { setTimeout(preview, 250); }
}
addEventListener('pagehide', () => {
 if (run?.session) fetch('/api/calibration', {method: 'POST', keepalive: true, headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: 'cancel', session_id: run.session})}).catch(() => {});
 if (previewURL) URL.revokeObjectURL(previewURL);
});
poll(); preview();
