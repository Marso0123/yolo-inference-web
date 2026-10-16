"use strict";
const $ = (id) => document.getElementById(id);
let kind = "image",
  state = {},
  files = [],
  busy = false,
  toastTimer,
  receivedGeneration = -1,
  receivedSeq = -1,
  socket;
let classModel = null,
  selectedClasses = new Set(),
  classSaveQueue = Promise.resolve(),
  classSavePending = 0,
  classSaveFailed = false,
  classSaveRevision = 0;
let batchFiles = [],
  batchId = null,
  batchUploaded = 0,
  batchUploading = false,
  batchUploadController = null,
  batchJobs = [];
const statusNames = {
  idle: "等待开始",
  ready: "模型就绪",
  loading: "加载中",
  running: "正在推理",
  finished: "推理完成",
  stopped: "已停止",
  stopping: "停止中",
  reconnecting: "正在重连",
  exporting: "正在转换",
  packaging: "正在打包",
  error: "发生错误",
};
function notify(message, error = false) {
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(
    () => ($("toast").hidden = true),
    error ? 9000 : 4500,
  );
}
async function api(path, data, method = "POST") {
  const options = { method };
  if (data !== undefined) {
    options.body = JSON.stringify(data);
    options.headers = { "Content-Type": "application/json" };
  }
  const response = await fetch("/api/" + path, options);
  const body = await response.json();
  if (!response.ok) {
    const error = Array.isArray(body.detail)
      ? body.detail.map((e) => e.msg).join("；")
      : body.detail;
    throw new Error(error || "操作失败");
  }
  return body;
}
async function action(fn) {
  if (busy) return;
  busy = true;
  document.body.classList.add("busy");
  controls();
  try {
    await fn();
  } catch (e) {
    notify(e.message, true);
  } finally {
    busy = false;
    document.body.classList.remove("busy");
    await poll();
  }
}
function options() {
  return {
    device: $("device").value,
    imgsz: Number($("imgsz").value),
    conf: Number($("conf").value),
    iou: Number($("iou").value),
    classes:
      classModel?.id === $("modelSelect").value
        ? [...selectedClasses].sort((a, b) => a - b)
        : [],
    boxes: $("boxes").checked,
    masks: $("masks").checked,
    labels: $("labels").checked,
  };
}
function validSize(value) {
  const n = Number(value);
  if (!Number.isInteger(n) || n < 32 || n > 2048 || n % 32)
    throw new Error("输入尺寸必须是 32～2048 之间的 32 的倍数。");
  return n;
}
function populate(id, items, placeholder) {
  const select = $(id),
    previous = select.value;
  select.replaceChildren(new Option(placeholder, ""));
  for (const item of items) select.add(new Option(item.name, item.id));
  if (items.some((i) => i.id === previous)) select.value = previous;
}
async function refreshFiles() {
  files = await api("assets", undefined, "GET");
  populate(
    "modelSelect",
    files.filter((f) => f.kind === "model"),
    "选择已上传的模型",
  );
  populate(
    "exportModel",
    files.filter(
      (f) => f.kind === "model" && f.name.toLowerCase().endsWith(".pt"),
    ),
    "选择已上传的 PT 模型",
  );
  refreshMedia();
  renderSaved();
  await refreshBatches();
}
function refreshMedia() {
  populate(
    "mediaSelect",
    files.filter((f) => f.kind === "media" && f.media_type === kind),
    kind === "image" ? "或选择已上传的图片" : "或选择已上传的视频",
  );
}
async function upload(file, type) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch("/api/upload/" + type, {
    method: "POST",
    body: form,
  });
  const item = await response.json();
  if (!response.ok) throw new Error(item.detail || "上传失败");
  await refreshFiles();
  return item;
}
function clearPreview() {
  $("canvas").hidden = true;
  $("placeholder").hidden = false;
  const context = $("canvas").getContext("2d");
  context.clearRect(0, 0, $("canvas").width, $("canvas").height);
}
function classSummary() {
  if (!classModel) {
    $("classSummary").textContent = "加载模型后选择类别";
    return;
  }
  const total = Object.keys(classModel.names).length;
  $("classSummary").textContent =
    selectedClasses.size === 0 || selectedClasses.size === total
      ? `输出全部 ${total} 类`
      : `仅输出所选 ${selectedClasses.size} / ${total} 类`;
}
function renderClassList() {
  const query = $("classSearch").value.trim().toLocaleLowerCase();
  const entries = classModel ? Object.entries(classModel.names) : [];
  const visible = entries.filter(([id, name]) =>
    `${id} ${name}`.toLocaleLowerCase().includes(query),
  );
  $("classList").replaceChildren(
    ...visible.map(([id, name]) => {
      const label = document.createElement("label");
      label.className = "class-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = id;
      checkbox.checked = selectedClasses.has(Number(id));
      checkbox.setAttribute("aria-label", `${id} · ${name}`);
      const number = document.createElement("span");
      number.className = "class-number";
      number.textContent = id;
      const text = document.createElement("span");
      text.className = "class-name";
      text.textContent = name;
      checkbox.onchange = () => {
        if (checkbox.checked) selectedClasses.add(Number(id));
        else selectedClasses.delete(Number(id));
        classSummary();
        saveClassSelection();
      };
      label.append(checkbox, number, text);
      return label;
    }),
  );
  $("classEmpty").hidden = !classModel || visible.length !== 0;
  classSummary();
}
function bindClassModel(model) {
  classModel = model;
  selectedClasses = new Set(model?.classes || []);
  classSaveFailed = false;
  $("classSearch").value = "";
  $("retryClasses").hidden = true;
  $("classSaveState").textContent = model
    ? selectedClasses.size > 0
      ? "已恢复此模型的类别选择。"
      : "选择自动保存；未勾选时默认输出全部类别。"
    : "请先加载模型，读取类别名称。";
  renderClassList();
}
function saveClassSelection() {
  if (!classModel) return;
  const modelId = classModel.id;
  const selection = [...selectedClasses].sort((a, b) => a - b);
  const revision = ++classSaveRevision;
  classSavePending++;
  $("classSaveState").textContent = "正在保存选择…";
  $("retryClasses").hidden = true;
  controls();
  // Serialize fast checkbox clicks so the final choice is the last disk write.
  classSaveQueue = classSaveQueue.then(async () => {
    try {
      await api(
        `models/${modelId}/class-filter`,
        { classes: selection },
        "PUT",
      );
      if (revision === classSaveRevision) {
        classSaveFailed = false;
        $("classSaveState").textContent = "已按模型保存；下次加载自动恢复。";
        $("retryClasses").hidden = true;
      }
    } catch (error) {
      if (revision === classSaveRevision) {
        classSaveFailed = true;
        $("classSaveState").textContent = "保存失败，请重试。";
        $("retryClasses").hidden = false;
        notify(error.message, true);
      }
    } finally {
      classSavePending--;
      controls();
    }
  });
}
function renderModel() {
  const model = state.model;
  const selected = $("modelSelect").value;
  if (model && model.id === selected) {
    $("taskBadge").textContent =
      model.task === "segment" ? "实例分割" : "目标检测";
    $("modelInfo").textContent =
      `${Object.keys(model.names).length} 个类别 · 输入 ${Array.isArray(model.imgsz) ? model.imgsz.join(" × ") : model.imgsz}${model.cuda_nodes ? " · CUDA 已验证" : ""}`;
    bindClassModel(model);
    $("nmsHint").textContent = model.end2end
      ? "此模型使用端到端输出，IoU 阈值不参与常规 NMS。"
      : "适用于使用 NMS 的模型。";
  } else {
    $("taskBadge").textContent = "待加载";
    $("modelInfo").textContent = "加载后自动读取模型类别与任务类型。";
    bindClassModel(null);
  }
}
function controls() {
  const running = ["running", "reconnecting", "stopping"].includes(
    state.status,
  );
  const locked =
    busy ||
    running ||
    ["loading", "exporting", "packaging"].includes(state.status) ||
    state.saving;
  const loaded =
    state.model &&
    state.model.id === $("modelSelect").value &&
    options().device === (state.model.device.startsWith("GPU") ? "gpu" : "cpu");
  $("loadModel").disabled =
    locked || classSavePending > 0 || !$("modelSelect").value;
  $("start").disabled =
    locked ||
    classSavePending > 0 ||
    classSaveFailed ||
    !loaded ||
    options().imgsz !== state.loadSize;
  $("stop").disabled = busy || !running;
  $("snapshot").disabled = busy || !state.has_frame;
  $("record").disabled =
    busy ||
    state.saving ||
    (!state.recording &&
      !(
        state.status === "running" &&
        ["video", "rtsp"].includes(state.source_kind) &&
        state.has_frame
      ));
  $("record").textContent = state.saving
    ? "正在保存…"
    : state.recording
      ? "■ 停止并保存"
      : "● 开始录制";
  $("record").classList.toggle("primary", !!state.recording);
  $("exportButton").disabled =
    locked || classSavePending > 0 || !$("exportModel").value;
  for (const id of [
    "modelSelect",
    "modelFile",
    "device",
    "imgsz",
    "mediaFile",
    "mediaSelect",
    "rtspUrl",
    "exportModel",
    "exportFile",
    "exportSize",
    "conf",
    "iou",
    "boxes",
    "masks",
    "labels",
    "batchFolder",
    "batchFiles",
    "batchRemap",
    "batchPreview",
  ])
    $(id).disabled =
      locked ||
      (classSavePending > 0 &&
        [
          "modelSelect",
          "modelFile",
          "device",
          "imgsz",
          "exportModel",
          "exportFile",
        ].includes(id));
  $("classFilter").disabled =
    locked || !classModel || classModel.id !== $("modelSelect").value;
  document.querySelectorAll(".source").forEach((e) => (e.disabled = locked));
  $("batchCancel").disabled = !(
    batchUploading ||
    (!busy && (running || state.status === "packaging"))
  );
  const job = state.batch;
  $("batchZip").disabled =
    locked ||
    !job?.output_path ||
    ["running", "uploading", "packaging"].includes(job.status) ||
    !!job.archive;
  document
    .querySelectorAll(".history-zip")
    .forEach((e) => (e.disabled = locked));
}
function render(stateIn) {
  const previous = state;
  state = { ...stateIn, loadSize: previous.loadSize };
  if (
    state.model &&
    (!previous.model || state.model.id !== previous.model.id)
  ) {
    if (state.model.id === $("modelSelect").value) {
      state.loadSize = Number($("imgsz").value);
    }
  }
  const matchingModel =
    state.model?.id === $("modelSelect").value ? state.model : null;
  if (classModel?.id !== matchingModel?.id) renderModel();
  $("stateLabel").textContent = statusNames[state.status] || state.status;
  $("stateDot").className = "state-dot " + state.status;
  $("statusMessage").textContent = state.message;
  $("actualDevice").textContent = state.model?.device || "—";
  $("fps").textContent =
    state.has_frame && ["video", "rtsp"].includes(state.source_kind)
      ? state.fps
      : "—";
  $("latency").textContent = state.has_frame ? state.inference_ms : "—";
  $("count").textContent = state.has_frame ? state.count : "—";
  $("recordingInfo").hidden = !state.recording;
  $("recordingInfo").textContent = state.recording
    ? `录制 ${Number(state.recording_fps || 25).toFixed(2)} FPS · ${state.recording_mode === "source_frames" ? "源视频逐帧保存，不跳帧" : "实时画面，流畅度取决于实际推理速度"}`
    : "";
  $("position").textContent = state.position
    ? `${state.position.seconds.toFixed(1)}s${state.position.duration ? " / " + state.position.duration.toFixed(1) + "s" : ""}`
    : "";
  $("counts").replaceChildren(
    ...Object.entries(state.counts || {}).map(([name, count]) => {
      const span = document.createElement("span");
      span.textContent = `${name} ${count}`;
      return span;
    }),
  );
  if (!state.has_frame) clearPreview();
  if (state.export) {
    $("exportState").textContent = state.message;
    if (
      state.export.status === "done" &&
      state.export.result.id !== previous.export?.result?.id
    ) {
      showExport(state.export.result);
      refreshFiles().catch((e) => notify(e.message, true));
    }
  }
  if (state.last_saved && state.last_saved.id !== previous.last_saved?.id) {
    showDownload(state.last_saved);
    refreshFiles().catch((e) => notify(e.message, true));
  }
  if (state.record_error && state.record_error !== previous.record_error) {
    notify("视频保存失败：" + state.record_error, true);
  }
  renderBatch();
  if (
    state.batch &&
    (state.batch.status !== previous.batch?.status ||
      state.batch.archive?.id !== previous.batch?.archive?.id)
  )
    refreshBatches().catch((e) => notify(e.message, true));
  controls();
}
function downloadLink(item, label) {
  const a = document.createElement("a");
  a.href = "/api/download/" + item.id;
  a.textContent = label || "下载";
  a.setAttribute("download", item.name);
  return a;
}
function showDownload(item) {
  $("latestDownload").hidden = false;
  $("latestDownload").replaceChildren(
    document.createTextNode("已保存：" + item.name + "　"),
    downloadLink(item, "下载文件"),
  );
}
function showExport(item) {
  const button = document.createElement("button");
  button.className = "button secondary";
  button.textContent = "加载这个 ONNX";
  button.onclick = () =>
    action(async () => {
      await refreshFiles();
      $("modelSelect").value = item.id;
      switchView("inference");
      renderModel();
      await loadModel();
    });
  $("exportResult").hidden = false;
  $("exportResult").replaceChildren(
    downloadLink(item, "下载 ONNX 模型"),
    document.createTextNode("　"),
    button,
  );
}
function renderSaved() {
  const items = files.filter((f) => f.kind === "output").reverse();
  $("savedList").replaceChildren();
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty-list";
    empty.textContent = "还没有保存记录。推理后点击保存图片，或手动录制视频。";
    $("savedList").append(empty);
    return;
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "saved-item";
    const info = document.createElement("div"),
      title = document.createElement("b"),
      small = document.createElement("small");
    title.textContent = item.name;
    small.textContent =
      item.output_type === "dataset"
        ? "YOLO 数据集 · ZIP · 原图与标签"
        : item.output_type === "video"
          ? `标注视频 · MP4 / H.264 · 无音频${item.fps ? ` · ${Number(item.fps).toFixed(2)} FPS` : ""}`
          : "标注图片 · JPG";
    info.append(title, small);
    row.append(info, downloadLink(item, "下载文件 ↗"));
    $("savedList").append(row);
  }
}
async function poll() {
  try {
    render(await api("status", undefined, "GET"));
  } catch (e) {
    $("statusMessage").textContent = "服务连接中断，请检查启动窗口";
  }
}
async function loadModel() {
  await classSaveQueue;
  validSize($("imgsz").value);
  const info = await api("load", {
    model_id: $("modelSelect").value,
    options: options(),
  });
  state.model = info;
  state.loadSize = Number($("imgsz").value);
  renderModel();
  notify("模型已加载，推理设备已检查");
}
function switchView(view) {
  const target = view === "batch" ? "inference" : view;
  document
    .querySelectorAll(".view")
    .forEach((e) => e.classList.toggle("active", e.id === target));
  document
    .querySelectorAll(".nav")
    .forEach((e) => e.classList.toggle("active", e.dataset.view === view));
}
document.querySelectorAll(".nav").forEach(
  (e) =>
    (e.onclick = () => {
      if (["batch", "inference"].includes(e.dataset.view)) {
        if (
          [
            "running",
            "reconnecting",
            "stopping",
            "packaging",
            "loading",
            "exporting",
          ].includes(state.status) ||
          busy
        ) {
          const desired = state.source_kind === "batch" ? "batch" : kind;
          document.querySelector(`.source[data-kind="${desired}"]`).onclick();
        } else
          document
            .querySelector(
              `.source[data-kind="${e.dataset.view === "batch" ? "batch" : kind === "batch" ? "image" : kind}"]`,
            )
            .click();
      } else switchView(e.dataset.view);
      if (e.dataset.view === "saved")
        refreshFiles().catch((err) => notify(err.message, true));
    }),
);
document.querySelectorAll(".source").forEach(
  (e) =>
    (e.onclick = () => {
      kind = e.dataset.kind;
      document
        .querySelectorAll(".source")
        .forEach((b) => b.classList.toggle("active", b === e));
      $("fileSource").hidden = ["rtsp", "batch"].includes(kind);
      $("rtspSource").hidden = kind !== "rtsp";
      $("batchSource").hidden = kind !== "batch";
      $("batchResult").hidden = kind !== "batch";
      document.body.classList.toggle("batch-mode", kind === "batch");
      $("inferenceTitle").textContent =
        kind === "batch" ? "批量推理，导出预标注。" : "让训练结果，看得见。";
      $("inferenceIntro").textContent =
        kind === "batch"
          ? "逐张处理图片，保存原图与 YOLO 格式标签；未检出图片列为待复核。"
          : "加载自己的 YOLO 模型，查看检测与实例分割效果。";
      $("start").textContent =
        kind === "batch" ? "▶ 开始批量推理" : "▶ 开始推理";
      switchView(kind === "batch" ? "batch" : "inference");
      $("mediaFile").accept =
        kind === "image"
          ? ".jpg,.jpeg,.png,.bmp,.webp"
          : ".mp4,.avi,.mov,.mkv,.m4v,.webm,.ts";
      $("uploadTitle").textContent =
        kind === "image" ? "选择待推理图片" : "选择待推理视频";
      $("mediaName").textContent =
        kind === "image"
          ? "JPG / PNG / BMP / WebP"
          : "MP4 / AVI / MOV / MKV / WebM / TS";
      $("sourceLabel").textContent =
        kind === "batch"
          ? "图片 · 批量预标注"
          : kind === "rtsp"
            ? "RTSP · 最新帧"
            : kind === "video"
              ? "视频 · 实时播放"
              : "图片 · 单次推理";
      refreshMedia();
      controls();
    }),
);
$("showUrl").onchange = () => {
  $("rtspUrl").type = $("showUrl").checked ? "text" : "password";
};
for (const id of ["conf", "iou"])
  $(id).oninput = () => {
    $(id + "Value").value = Number($(id).value).toFixed(2);
  };
$("modelSelect").onchange = () => {
  renderModel();
  controls();
};
$("device").onchange = controls;
$("imgsz").oninput = controls;
$("exportModel").onchange = controls;
$("classSearch").oninput = renderClassList;
$("resetClasses").onclick = () => {
  selectedClasses.clear();
  renderClassList();
  saveClassSelection();
};
$("retryClasses").onclick = saveClassSelection;
for (const [id, type] of [
  ["modelFile", "model"],
  ["exportFile", "model"],
  ["mediaFile", "media"],
])
  $(id).onchange = () =>
    action(async () => {
      const file = $(id).files[0];
      if (!file) return;
      notify("正在上传 " + file.name + "…");
      const item = await upload(file, type);
      if (type === "model") {
        $("modelSelect").value = item.id;
        if (item.name.toLowerCase().endsWith(".pt"))
          $("exportModel").value = item.id;
        renderModel();
      } else {
        $("mediaSelect").value = item.id;
        $("mediaName").textContent = item.name;
      }
      $(id).value = "";
      notify("文件已上传");
    });
$("loadModel").onclick = () => action(loadModel);
$("start").onclick = () =>
  action(async () => {
    if (kind === "batch") {
      await startBatch();
      return;
    }
    if (kind !== "rtsp" && !$("mediaSelect").value)
      throw new Error("请先选择图片或视频。");
    if (kind === "rtsp" && !$("rtspUrl").value.trim())
      throw new Error("请输入 RTSP 地址。");
    validSize($("imgsz").value);
    clearPreview();
    const result = await api("start", {
      model_id: $("modelSelect").value,
      options: options(),
      kind,
      media_id: $("mediaSelect").value || null,
      url: kind === "rtsp" ? $("rtspUrl").value : "",
    });
    render(result);
  });
$("stop").onclick = () =>
  action(async () => {
    await api("stop");
    notify("推理已停止");
  });
$("snapshot").onclick = () =>
  action(async () => {
    const item = await api("save-image");
    showDownload(item);
    downloadLink(item).click();
    notify("标注图片已保存");
    await refreshFiles();
  });
$("record").onclick = () =>
  action(async () => {
    if (state.recording) {
      notify("正在编码保存视频…");
      const item = await api("record/stop");
      showDownload(item);
      downloadLink(item).click();
      await refreshFiles();
      notify("标注视频已保存");
    } else {
      await api("record/start");
      notify(
        state.source_kind === "video"
          ? "已开始逐帧录制，保存时保持源视频帧率；再次点击停止并保存"
          : "实时录制已开始，再次点击即可停止并保存",
      );
    }
  });
$("exportButton").onclick = () =>
  action(async () => {
    validSize($("exportSize").value);
    $("exportResult").hidden = true;
    await api("export", {
      model_id: $("exportModel").value,
      imgsz: Number($("exportSize").value),
    });
    notify("转换已开始");
  });
$("refreshSaved").onclick = () => action(refreshFiles);

for (const id of ["batchFolder", "batchFiles"])
  $(id).onchange = () => {
    const selected = [...$(id).files];
    batchFiles = selected.filter((f) =>
      /\.(jpe?g|png|bmp|webp)$/i.test(f.name),
    );
    batchId = null;
    batchUploaded = 0;
    $("batchSelection").textContent =
      `已选择 ${batchFiles.length} 张图片${selected.length > batchFiles.length ? `；忽略 ${selected.length - batchFiles.length} 个非图片文件` : ""}。`;
    $(id).value = "";
  };
async function startBatch() {
  if (!batchFiles.length) throw new Error("请先选择图片文件夹或多张图片。");
  await classSaveQueue;
  validSize($("imgsz").value);
  if (!batchId) batchId = (await api("batches")).id;
  batchUploading = true;
  batchUploadController = new AbortController();
  $("batchTask").textContent = "正在上传";
  $("batchOutput").textContent = "";
  $("batchDownload").replaceChildren();
  controls();
  try {
    for (; batchUploaded < batchFiles.length; batchUploaded++) {
      const file = batchFiles[batchUploaded];
      $("batchProgressText").textContent =
        `上传图片 ${batchUploaded + 1} / ${batchFiles.length}；尚未开始推理`;
      $("batchCurrent").textContent = file.webkitRelativePath || file.name;
      $("batchProgress").max = batchFiles.length;
      $("batchProgress").value = batchUploaded;
      const form = new FormData();
      form.append("file", file);
      form.append("relative_path", file.webkitRelativePath || file.name);
      form.append("upload_id", String(batchUploaded));
      const response = await fetch(`/api/batches/${batchId}/upload`, {
        method: "POST",
        body: form,
        signal: batchUploadController.signal,
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || "批量上传失败");
    }
    if (batchUploadController.signal.aborted) return;
    clearPreview();
    render(
      await api("batch/start", {
        model_id: $("modelSelect").value,
        options: options(),
        batch_id: batchId,
        remap: $("batchRemap").checked,
        preview: $("batchPreview").checked,
      }),
    );
    batchId = null;
    batchUploaded = 0;
    notify("批量任务已开始，结果保存在新的数据集目录。");
  } catch (e) {
    if (e.name === "AbortError")
      notify("上传已取消；已上传图片保留，再次开始可继续上传。");
    else throw e;
  } finally {
    batchUploading = false;
    batchUploadController = null;
    renderBatch();
  }
}
const batchStatusNames = {
  uploading: "待上传／待开始",
  running: "正在推理",
  completed: "已完成",
  completed_with_errors: "完成，有失败项",
  cancelled: "已取消",
  failed: "失败",
  packaging: "正在打包",
  interrupted: "服务中断",
};
function renderBatch() {
  if (batchUploading) return;
  const job = state.batch;
  if (!job) return;
  $("batchTask").textContent = batchStatusNames[job.status] || job.status;
  $("batchProgress").max = job.total || 1;
  $("batchProgress").value = job.done + job.failed;
  $("batchProgressText").textContent =
    `${job.done + job.failed} / ${job.total} 张已处理 · 保存 ${job.done} · 失败 ${job.failed} · 待复核 ${job.review}`;
  $("batchCurrent").textContent = job.current
    ? "当前：" + job.current
    : job.message;
  $("batchOutput").textContent = job.output_path
    ? "数据集目录：" + job.output_path
    : "";
  $("batchDownload").replaceChildren(
    ...(job.archive ? [downloadLink(job.archive, "下载数据集 ZIP ↗")] : []),
  );
}
async function refreshBatches() {
  batchJobs = await api("batches", undefined, "GET");
  $("batchHistory").replaceChildren();
  for (const job of batchJobs.slice(0, 12)) {
    const row = document.createElement("div"),
      text = document.createElement("span");
    row.className = "batch-history-row";
    text.textContent = `${job.created} · ${batchStatusNames[job.status] || job.status} · 保存 ${job.done} / ${job.total} · 待复核 ${job.review}`;
    text.title = job.output_path || job.message;
    row.append(text);
    if (job.archive) row.append(downloadLink(job.archive, "下载 ZIP"));
    else if (
      job.output_path &&
      !["running", "packaging"].includes(job.status)
    ) {
      const button = document.createElement("button");
      button.className = "class-reset history-zip";
      button.textContent = "生成 ZIP";
      button.onclick = () =>
        action(async () => render(await api(`batches/${job.id}/archive`)));
      row.append(button);
    }
    $("batchHistory").append(row);
  }
  if (!batchJobs.length) $("batchHistory").textContent = "尚无批量任务。";
  controls();
}
$("batchCancel").onclick = () => {
  if (batchUploading) batchUploadController?.abort();
  else
    action(async () => {
      await api("stop");
      notify("任务已取消，已保存结果保留。");
      await refreshBatches();
    });
};
$("batchZip").onclick = () =>
  action(async () => render(await api(`batches/${state.batch.id}/archive`)));
function connect() {
  socket = new WebSocket(
    (location.protocol === "https:" ? "wss://" : "ws://") +
      location.host +
      "/ws",
  );
  socket.binaryType = "arraybuffer";
  socket.onmessage = async (e) => {
    try {
      const data = new Uint8Array(e.data),
        length = new DataView(e.data).getUint32(0, true),
        metadata = JSON.parse(
          new TextDecoder().decode(data.subarray(4, 4 + length)),
        );
      if (
        metadata.generation < state.generation ||
        metadata.generation < receivedGeneration ||
        (metadata.generation === receivedGeneration &&
          metadata.seq <= receivedSeq)
      )
        return;
      receivedGeneration = metadata.generation;
      receivedSeq = metadata.seq;
      const bitmap = await createImageBitmap(
        new Blob([data.subarray(4 + length)], { type: "image/jpeg" }),
      );
      if (
        metadata.generation !== receivedGeneration ||
        metadata.seq !== receivedSeq ||
        metadata.generation < state.generation ||
        (metadata.generation === state.generation && metadata.seq < state.seq)
      ) {
        bitmap.close();
        return;
      }
      const canvas = $("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext("2d").drawImage(bitmap, 0, 0);
      bitmap.close();
      canvas.hidden = false;
      $("placeholder").hidden = true;
    } catch (e) {
      notify("预览帧解析失败：" + e.message, true);
    }
  };
  socket.onclose = () => {
    receivedGeneration = -1;
    receivedSeq = -1;
    setTimeout(connect, 2000);
  };
}
(async () => {
  try {
    const hw = await api("hardware", undefined, "GET");
    $("hardware").textContent = hw.gpu_available
      ? hw.gpu_name
      : "CPU 模式 · 未检测到 CUDA";
    if (!hw.gpu_available) $("device").value = "cpu";
    await refreshFiles();
    await poll();
    connect();
    setInterval(poll, 650);
  } catch (e) {
    notify("服务初始化失败：" + e.message, true);
  }
})();
