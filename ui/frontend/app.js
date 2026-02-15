const STORAGE_KEY_V2 = "anima_ui_state_v2";
const STORAGE_KEYS_LEGACY = ["anima_ui_state_v1", "anima_ui_state"];

const DEFAULT_SCHEMA = {
  version: 2,
  steps: [
    { id: "model", sections: ["model", "precision_opt", "format"], required: ["model.pretrained_model_name_or_path", "model.qwen3", "model.vae"] },
    { id: "dataset", sections: ["dataset", "advanced"], required: ["dataset.dataset_path"] },
    { id: "trainer", sections: ["trainer", "saving", "resume", "sample"], required: ["saving.output_dir", "saving.output_name"] },
    { id: "execute", sections: ["execute"], required: [] }
  ]
};

const state = {
  lang: "zh-TW",
  layout_mode: "dashboard",
  theme_preference: "system",
  resolved_theme: "light",
  i18n: {},
  defaults: { fields: {} },
  defaultsWarning: "",
  schema: DEFAULT_SCHEMA,
  profiles: [],
  activeProfileId: null,
  activeStep: "model",
  activeSection: "model",
  recentPaths: [],
  status: null,
  summary: null,
  lastMessage: "",
  lastMessageClass: "",
  loading: { validate: false, start: false, stop: false, startTb: false, stopTb: false, import: false },
  eventSource: null,
  mediaQueryDark: null,
  pollingTimer: null,
  saveTimer: null
};

const el = {
  appTitle: document.getElementById("app-title"),
  defaultsWarning: document.getElementById("defaults-warning"),
  themeLabel: document.getElementById("theme-label"),
  themeSelect: document.getElementById("theme-select"),
  langLabel: document.getElementById("lang-label"),
  langSelect: document.getElementById("lang-select"),
  addProfileBtn: document.getElementById("add-profile-btn"),
  duplicateProfileBtn: document.getElementById("duplicate-profile-btn"),
  deleteProfileBtn: document.getElementById("delete-profile-btn"),
  exportBtn: document.getElementById("export-btn"),
  importBtn: document.getElementById("import-btn"),
  importFile: document.getElementById("import-file"),
  stepsTitle: document.getElementById("steps-title"),
  stepNav: document.getElementById("step-nav"),
  profileTabs: document.getElementById("profile-tabs"),
  profileNameInput: document.getElementById("profile-name-input"),
  runEnabledInput: document.getElementById("run-enabled-input"),
  labelProfileName: document.getElementById("label-profile-name"),
  labelRunEnabled: document.getElementById("label-run-enabled"),
  sectionTabs: document.getElementById("section-tabs"),
  sectionHelp: document.getElementById("section-help"),
  sectionContent: document.getElementById("section-content"),
  actionValidate: document.getElementById("action-validate"),
  actionStart: document.getElementById("action-start"),
  actionStop: document.getElementById("action-stop"),
  actionStartTb: document.getElementById("action-start-tb"),
  actionStopTb: document.getElementById("action-stop-tb"),
  statusTitle: document.getElementById("status-title"),
  cardBatchStateLabel: document.getElementById("card-batch-state-label"),
  cardBatchState: document.getElementById("card-batch-state"),
  cardCurrentJobLabel: document.getElementById("card-current-job-label"),
  cardCurrentJob: document.getElementById("card-current-job"),
  cardQueueLabel: document.getElementById("card-queue-label"),
  cardQueueCount: document.getElementById("card-queue-count"),
  cardTotalLabel: document.getElementById("card-total-label"),
  cardTotalCount: document.getElementById("card-total-count"),
  cardLastErrorLabel: document.getElementById("card-last-error-label"),
  cardLastError: document.getElementById("card-last-error"),
  cardTbLabel: document.getElementById("card-tb-label"),
  cardTbStatus: document.getElementById("card-tb-status"),
  jobsTitle: document.getElementById("jobs-title"),
  statusJobs: document.getElementById("status-jobs"),
  logPreviewTitle: document.getElementById("log-preview-title"),
  logPreview: document.getElementById("log-preview"),
  messages: document.getElementById("messages")
};

function t(key, fallback = "") {
  const parts = key.split(".");
  let obj = state.i18n;
  for (const part of parts) {
    if (!obj || !(part in obj)) return fallback || key;
    obj = obj[part];
  }
  return typeof obj === "string" ? obj : fallback || key;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function isZhLocale() {
  return String(state.lang || "").toLowerCase() === "zh-tw";
}

function taskDefaultLabel() {
  return t("task_default_name", isZhLocale() ? "任務" : "Task");
}

function taskCopySuffix() {
  return t("task_copy_suffix", isZhLocale() ? "副本" : "Copy");
}

function normalizeThemePreference(value) {
  return value === "light" || value === "dark" || value === "system" ? value : "system";
}

function resolveTheme(preference) {
  if (preference === "dark" || preference === "light") return preference;
  const isDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  return isDark ? "dark" : "light";
}

function applyTheme() {
  state.resolved_theme = resolveTheme(state.theme_preference);
  document.documentElement.setAttribute("data-theme", state.resolved_theme);
}

function bindThemeEvents() {
  if (!window.matchMedia) return;
  if (!state.mediaQueryDark) {
    state.mediaQueryDark = window.matchMedia("(prefers-color-scheme: dark)");
  }
  const onChange = () => {
    if (state.theme_preference !== "system") return;
    applyTheme();
  };
  if (typeof state.mediaQueryDark.addEventListener === "function") {
    state.mediaQueryDark.addEventListener("change", onChange);
  } else if (typeof state.mediaQueryDark.addListener === "function") {
    state.mediaQueryDark.addListener(onChange);
  }
}

function uuid() {
  return `${Math.random().toString(16).slice(2)}${Date.now().toString(16)}`;
}

function deepMerge(target, source) {
  if (!source || typeof source !== "object") return target;
  for (const key of Object.keys(source)) {
    const src = source[key];
    if (src && typeof src === "object" && !Array.isArray(src) && target[key] && typeof target[key] === "object" && !Array.isArray(target[key])) {
      deepMerge(target[key], src);
    } else {
      target[key] = src;
    }
  }
  return target;
}

function getDefault(dest, fallback = null) {
  const field = state.defaults.fields?.[dest] || {};
  return field.default ?? fallback;
}

function getChoices(dest, fallback = []) {
  const field = state.defaults.fields?.[dest] || {};
  const choices = Array.isArray(field.choices) ? field.choices.filter((x) => x !== null) : fallback;
  return choices.length > 0 ? choices : fallback;
}

function createDefaultProfile(index = 1) {
  const savePrecisionChoices = getChoices("save_precision", ["fp16", "bf16", "float"]);
  const mixedChoices = getChoices("mixed_precision", ["no", "fp16", "bf16"]);
  const schedulerChoices = getChoices("lr_scheduler", ["constant", "cosine", "linear"]);
  const samplerChoices = getChoices("sample_sampler", ["ddim", "euler", "euler_a"]);
  const timestepChoices = getChoices("timestep_sampling", ["sigma", "uniform", "sigmoid", "shift", "flux_shift"]);
  const now = Date.now() / 1000;

  return {
    id: uuid(),
    name: `${taskDefaultLabel()} ${index}`,
    run_enabled: true,
    model: { pretrained_model_name_or_path: "", qwen3: "", vae: "", llm_adapter_path: "", t5_tokenizer_path: "" },
    precision_opt: {
      mixed_precision: getDefault("mixed_precision", mixedChoices[0] || "bf16"),
      save_precision: getDefault("save_precision", savePrecisionChoices[0] || "bf16"),
      xformers: false,
      attn_mode: getDefault("attn_mode", "torch"),
      split_attn: false,
      gradient_checkpointing: !!getDefault("gradient_checkpointing", true),
      blocks_to_swap: "",
      cache_latents: !!getDefault("cache_latents", false),
      cache_latents_to_disk: !!getDefault("cache_latents_to_disk", false),
      cache_text_encoder_outputs: !!getDefault("cache_text_encoder_outputs", false),
      cache_text_encoder_outputs_to_disk: !!getDefault("cache_text_encoder_outputs_to_disk", false)
    },
    dataset: {
      dataset_path: "",
      reg_dataset_path: "",
      repeats: 1,
      caption_extension: ".txt",
      color_aug: false,
      flip_aug: false,
      random_crop: false,
      shuffle_caption: false,
      caption_dropout_rate: 0,
      caption_tag_dropout_rate: 0,
      token_warmup_step: 0,
      inspect_result: null
    },
    trainer: {
      learning_rate: getDefault("learning_rate", 1),
      lr_scheduler: getDefault("lr_scheduler", schedulerChoices[0] || "cosine"),
      lr_warmup_steps: getDefault("lr_warmup_steps", 300),
      optimizer_type: getDefault("optimizer_type", "AdamW"),
      optimizer_args: [],
      text_encoder_lr: getDefault("text_encoder_lr", 0),
      llm_adapter_lr: getDefault("llm_adapter_lr", 0),
      train_batch_size: getDefault("train_batch_size", 1),
      max_train_steps: getDefault("max_train_steps", 3000),
      max_train_epochs: getDefault("max_train_epochs", ""),
      max_grad_norm: getDefault("max_grad_norm", 1)
    },
    advanced: {
      timestep_sampling: getDefault("timestep_sampling", timestepChoices[2] || "sigmoid"),
      discrete_flow_shift: getDefault("discrete_flow_shift", 1),
      sigmoid_scale: getDefault("sigmoid_scale", 1),
      noise_offset: "",
      ip_noise_gamma: "",
      min_timestep: "",
      max_timestep: ""
    },
    format: {
      model_type: "LyCORIS",
      network_module: "",
      network_dim: getDefault("network_dim", 32),
      network_alpha: getDefault("network_alpha", 1),
      accus_epochs: "",
      batch_size: getDefault("train_batch_size", 1),
      network_train_unet_only: !!getDefault("network_train_unet_only", true),
      network_args: []
    },
    saving: {
      output_dir: "",
      logging_dir: "",
      output_name: "",
      save_model_as: getDefault("save_model_as", "safetensors"),
      save_every_n_steps: getDefault("save_every_n_steps", 250),
      save_every_n_epochs: "",
      save_last_n_steps: "",
      save_last_n_epochs: ""
    },
    resume: { enabled: false, path: "" },
    sample: {
      enabled: false,
      sample_every_n_steps: "",
      sample_every_n_epochs: "",
      sample_sampler: getDefault("sample_sampler", samplerChoices[0] || "ddim"),
      sample_steps: 20,
      cfg_scale: 7,
      prompt: "",
      negative_prompt: "",
      width: 1024,
      height: 1024,
      seed: ""
    },
    raw_args: [],
    execute: { tb_logdir: "", tb_port: 6006, validation: null },
    meta: {
      created_at: now,
      updated_at: now,
      validation_state: { errors: [], warnings: [], field_errors: {}, field_warnings: {}, valid: false },
      last_run_snapshot: null
    }
  };
}

function normalizeProfile(profile, index) {
  const merged = deepMerge(createDefaultProfile(index), profile && typeof profile === "object" ? profile : {});
  merged.id = String(merged.id || uuid());
  merged.name = String(merged.name || `Task ${index}`);
  merged.run_enabled = !!merged.run_enabled;
  merged.meta = merged.meta || {};
  merged.meta.updated_at = Date.now() / 1000;
  merged.meta.created_at = Number(merged.meta.created_at || merged.meta.updated_at);
  merged.meta.validation_state = merged.meta.validation_state || { errors: [], warnings: [], field_errors: {}, field_warnings: {}, valid: false };
  return merged;
}

function normalizeState(raw) {
  const source = raw?.ui_state && typeof raw.ui_state === "object" ? raw.ui_state : raw;
  const profilesRaw = Array.isArray(source?.profiles) ? source.profiles : [];
  const profiles = profilesRaw.map((p, idx) => normalizeProfile(p, idx + 1));
  if (!profiles.length) profiles.push(createDefaultProfile(1));

  const ids = new Set(profiles.map((p) => p.id));
  const activeProfileId = ids.has(source?.active_profile_id) ? source.active_profile_id : profiles[0].id;
  const knownSteps = new Set((state.schema?.steps || DEFAULT_SCHEMA.steps).map((x) => x.id));
  const activeStep = knownSteps.has(source?.active_step) ? source.active_step : "model";

  return {
    version: 2,
    lang: source?.lang || state.lang,
    layout_mode: source?.layout_mode || "dashboard",
    theme_preference: normalizeThemePreference(source?.theme_preference),
    active_profile_id: activeProfileId,
    active_step: activeStep,
    profiles,
    recent_paths: Array.isArray(source?.recent_paths) ? source.recent_paths.filter((x) => typeof x === "string") : []
  };
}

function currentProfile() {
  return state.profiles.find((p) => p.id === state.activeProfileId) || null;
}

function setByPath(obj, path, value) {
  const parts = path.split(".");
  let ref = obj;
  for (let i = 0; i < parts.length - 1; i += 1) {
    if (!(parts[i] in ref)) ref[parts[i]] = {};
    ref = ref[parts[i]];
  }
  ref[parts[parts.length - 1]] = value;
}

function getByPath(obj, path, fallback = "") {
  const parts = path.split(".");
  let ref = obj;
  for (const key of parts) {
    if (!ref || !(key in ref)) return fallback;
    ref = ref[key];
  }
  return ref;
}

function sectionsForStep(stepId) {
  const step = (state.schema?.steps || DEFAULT_SCHEMA.steps).find((item) => item.id === stepId);
  return step ? step.sections : ["model"];
}

function stepForPath(path) {
  if (!path) return state.activeStep;
  if (path === "raw_args") return "execute";
  const section = path.split(".")[0];
  const steps = state.schema?.steps || DEFAULT_SCHEMA.steps;
  for (const step of steps) {
    if (step.sections.includes(section)) return step.id;
  }
  return state.activeStep;
}

function isPathInStep(path, stepId) {
  const sections = sectionsForStep(stepId);
  if (path === "raw_args") return sections.includes("execute");
  return sections.includes(path.split(".")[0]);
}

function getValidation(profile) {
  return profile?.meta?.validation_state || profile?.execute?.validation || { errors: [], warnings: [], field_errors: {}, field_warnings: {}, valid: false };
}

function pathHasValue(profile, path) {
  const value = getByPath(profile, path, "");
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim().length > 0;
  return true;
}

function computeStepState(profile, stepId) {
  const validation = getValidation(profile);
  const hasError = Object.keys(validation.field_errors || {}).some((path) => isPathInStep(path, stepId));
  if (hasError) return "error";

  const step = (state.schema?.steps || DEFAULT_SCHEMA.steps).find((x) => x.id === stepId);
  const requiredMissing = (step?.required || []).some((path) => !pathHasValue(profile, path));
  if (requiredMissing) return "warning";

  const hasWarning = Object.keys(validation.field_warnings || {}).some((path) => isPathInStep(path, stepId));
  return hasWarning ? "warning" : "complete";
}

function fieldHelp(path, labelKey) {
  const custom = t(`field_help.${labelKey}`, "");
  if (custom && custom !== `field_help.${labelKey}`) return custom;
  if (isZhLocale()) return "";
  const dest = path.split(".").slice(-1)[0];
  const parserHelp = state.defaults.fields?.[dest]?.help;
  return typeof parserHelp === "string" ? parserHelp : "";
}

function getFieldIssues(path) {
  const v = getValidation(currentProfile());
  return { errors: v.field_errors?.[path] || [], warnings: v.field_warnings?.[path] || [] };
}

function fieldHtml({ path, labelKey, type = "text", help = "", choices = null, full = false, placeholder = "", disabled = false }) {
  const profile = currentProfile();
  const value = profile ? getByPath(profile, path, "") : "";
  const issues = getFieldIssues(path);
  const issueClass = issues.errors.length ? "has-error" : issues.warnings.length ? "has-warning" : "";
  const className = `${full ? "field full" : "field"} ${issueClass}`.trim();
  const helpText = help || fieldHelp(path, labelKey);

  let input = "";
  if (type === "checkbox") {
    input = `<input data-path="${escapeHtml(path)}" type="checkbox" ${value ? "checked" : ""} ${disabled ? "disabled" : ""} />`;
  } else if (type === "textarea") {
    input = `<textarea data-path="${escapeHtml(path)}" ${disabled ? "disabled" : ""} placeholder="${escapeHtml(placeholder)}">${escapeHtml(value)}</textarea>`;
  } else if (type === "select") {
    const opts = (choices || []).map((c) => `<option value="${escapeHtml(c)}" ${String(value) === String(c) ? "selected" : ""}>${escapeHtml(c)}</option>`).join("");
    input = `<select data-path="${escapeHtml(path)}" ${disabled ? "disabled" : ""}>${opts}</select>`;
  } else {
    input = `<input data-path="${escapeHtml(path)}" type="${type}" value="${escapeHtml(value)}" ${disabled ? "disabled" : ""} placeholder="${escapeHtml(placeholder)}" />`;
  }

  const issuesHtml = [
    ...issues.errors.map((msg) => `<div class="issue error">${escapeHtml(msg)}</div>`),
    ...issues.warnings.map((msg) => `<div class="issue warning">${escapeHtml(msg)}</div>`)
  ].join("");

  return `<div class="${className}"><label>${escapeHtml(t(`fields.${labelKey}`))}</label>${input}${helpText ? `<small>${escapeHtml(helpText)}</small>` : ""}${issuesHtml}</div>`;
}

function renderArgList(path, labelKey) {
  const profile = currentProfile();
  const values = Array.isArray(getByPath(profile, path, [])) ? getByPath(profile, path, []) : [];
  const rows = values.map((arg, idx) => `<div class="arg-row"><input type="text" data-arg-path="${escapeHtml(path)}" data-arg-index="${idx}" value="${escapeHtml(arg)}" /><button type="button" data-arg-remove="${escapeHtml(path)}" data-arg-index="${idx}" class="arg-remove">x</button></div>`).join("");
  return `<div class="field full"><label>${escapeHtml(t(`fields.${labelKey}`))}</label><div class="arg-list">${rows || ""}</div><div class="button-row" style="margin-top:8px;"><button type="button" data-arg-add="${escapeHtml(path)}">${escapeHtml(t("fields.add_arg"))}</button></div></div>`;
}

function renderDatasetInspection() {
  const result = currentProfile()?.dataset?.inspect_result;
  return `<div class="field full"><label>${escapeHtml(t("fields.inspect_result"))}</label><pre class="status-output" style="min-height:120px;">${escapeHtml(result ? JSON.stringify(result, null, 2) : "")}</pre></div>`;
}
function renderSection() {
  const profile = currentProfile();
  if (!profile) return;

  const schedulerChoices = getChoices("lr_scheduler", ["constant", "cosine", "linear"]);
  const samplerChoices = getChoices("sample_sampler", ["ddim", "euler", "euler_a"]);
  const mixedChoices = getChoices("mixed_precision", ["no", "fp16", "bf16"]);
  const savePrecisionChoices = getChoices("save_precision", ["float", "fp16", "bf16"]);
  const attnChoices = getChoices("attn_mode", ["torch", "xformers", "flash", "sageattn", "sdpa"]);
  const timestepChoices = getChoices("timestep_sampling", ["sigma", "uniform", "sigmoid", "shift", "flux_shift"]);

  const cacheOn = profile.precision_opt.cache_latents || profile.precision_opt.cache_text_encoder_outputs;
  const disableAug = cacheOn;

  let html = "";
  switch (state.activeSection) {
    case "model":
      html += fieldHtml({ path: "model.pretrained_model_name_or_path", labelKey: "pretrained", full: true });
      html += fieldHtml({ path: "model.qwen3", labelKey: "qwen3", full: true });
      html += fieldHtml({ path: "model.vae", labelKey: "vae", full: true });
      html += fieldHtml({ path: "model.llm_adapter_path", labelKey: "llm_adapter_path" });
      html += fieldHtml({ path: "model.t5_tokenizer_path", labelKey: "t5_tokenizer_path" });
      break;
    case "precision_opt":
      html += fieldHtml({ path: "precision_opt.mixed_precision", labelKey: "mixed_precision", type: "select", choices: mixedChoices });
      html += fieldHtml({ path: "precision_opt.save_precision", labelKey: "save_precision", type: "select", choices: savePrecisionChoices });
      html += fieldHtml({ path: "precision_opt.attn_mode", labelKey: "attn_mode", type: "select", choices: attnChoices });
      html += fieldHtml({ path: "precision_opt.blocks_to_swap", labelKey: "blocks_to_swap", type: "number" });
      html += fieldHtml({ path: "precision_opt.xformers", labelKey: "xformers", type: "checkbox" });
      html += fieldHtml({ path: "precision_opt.split_attn", labelKey: "split_attn", type: "checkbox" });
      html += fieldHtml({ path: "precision_opt.gradient_checkpointing", labelKey: "gradient_checkpointing", type: "checkbox" });
      html += fieldHtml({ path: "precision_opt.cache_latents", labelKey: "cache_latents", type: "checkbox" });
      html += fieldHtml({ path: "precision_opt.cache_latents_to_disk", labelKey: "cache_latents_to_disk", type: "checkbox" });
      html += fieldHtml({ path: "precision_opt.cache_text_encoder_outputs", labelKey: "cache_text_encoder_outputs", type: "checkbox" });
      html += fieldHtml({ path: "precision_opt.cache_text_encoder_outputs_to_disk", labelKey: "cache_text_encoder_outputs_to_disk", type: "checkbox" });
      if (cacheOn) html += `<div class="field full"><span class="cache-warning">${escapeHtml(t("fields.cache_warning"))}</span></div>`;
      break;
    case "dataset":
      html += fieldHtml({ path: "dataset.dataset_path", labelKey: "dataset_path", full: true });
      html += fieldHtml({ path: "dataset.reg_dataset_path", labelKey: "reg_dataset_path", full: true });
      html += fieldHtml({ path: "dataset.repeats", labelKey: "repeats", type: "number" });
      html += fieldHtml({ path: "dataset.caption_extension", labelKey: "caption_extension" });
      html += fieldHtml({ path: "dataset.color_aug", labelKey: "color_aug", type: "checkbox", disabled: disableAug });
      html += fieldHtml({ path: "dataset.flip_aug", labelKey: "flip_aug", type: "checkbox", disabled: disableAug });
      html += fieldHtml({ path: "dataset.random_crop", labelKey: "random_crop", type: "checkbox", disabled: disableAug });
      html += fieldHtml({ path: "dataset.shuffle_caption", labelKey: "shuffle_caption", type: "checkbox", disabled: disableAug });
      html += fieldHtml({ path: "dataset.caption_dropout_rate", labelKey: "caption_dropout_rate", type: "number", disabled: disableAug });
      html += fieldHtml({ path: "dataset.caption_tag_dropout_rate", labelKey: "caption_tag_dropout_rate", type: "number", disabled: disableAug });
      html += fieldHtml({ path: "dataset.token_warmup_step", labelKey: "token_warmup_step", type: "number", disabled: disableAug });
      html += `<div class="field full"><button id="btn-inspect-dataset">${escapeHtml(t("fields.inspect_dataset"))}</button></div>`;
      html += renderDatasetInspection();
      break;
    case "trainer":
      html += fieldHtml({ path: "trainer.learning_rate", labelKey: "learning_rate", type: "number" });
      html += fieldHtml({ path: "trainer.lr_scheduler", labelKey: "lr_scheduler", type: "select", choices: schedulerChoices });
      html += fieldHtml({ path: "trainer.lr_warmup_steps", labelKey: "lr_warmup_steps", type: "number" });
      html += fieldHtml({ path: "trainer.optimizer_type", labelKey: "optimizer_type" });
      html += fieldHtml({ path: "trainer.text_encoder_lr", labelKey: "text_encoder_lr", type: "number" });
      html += fieldHtml({ path: "trainer.llm_adapter_lr", labelKey: "llm_adapter_lr", type: "number" });
      html += fieldHtml({ path: "trainer.train_batch_size", labelKey: "train_batch_size", type: "number" });
      html += fieldHtml({ path: "trainer.max_train_steps", labelKey: "max_train_steps", type: "number" });
      html += fieldHtml({ path: "trainer.max_train_epochs", labelKey: "max_train_epochs", type: "number" });
      html += fieldHtml({ path: "trainer.max_grad_norm", labelKey: "max_grad_norm", type: "number" });
      html += renderArgList("trainer.optimizer_args", "optimizer_args");
      break;
    case "advanced":
      html += fieldHtml({ path: "advanced.timestep_sampling", labelKey: "timestep_sampling", type: "select", choices: timestepChoices });
      html += fieldHtml({ path: "advanced.discrete_flow_shift", labelKey: "discrete_flow_shift", type: "number" });
      html += fieldHtml({ path: "advanced.sigmoid_scale", labelKey: "sigmoid_scale", type: "number" });
      html += fieldHtml({ path: "advanced.noise_offset", labelKey: "noise_offset", type: "number" });
      html += fieldHtml({ path: "advanced.ip_noise_gamma", labelKey: "ip_noise_gamma", type: "number" });
      html += fieldHtml({ path: "advanced.min_timestep", labelKey: "min_timestep", type: "number" });
      html += fieldHtml({ path: "advanced.max_timestep", labelKey: "max_timestep", type: "number" });
      break;
    case "format":
      html += fieldHtml({ path: "format.model_type", labelKey: "model_type", type: "select", choices: ["LoRA", "LyCORIS"] });
      html += fieldHtml({ path: "format.network_module", labelKey: "network_module" });
      html += fieldHtml({ path: "format.network_dim", labelKey: "network_dim", type: "number" });
      html += fieldHtml({ path: "format.network_alpha", labelKey: "network_alpha", type: "number" });
      html += fieldHtml({ path: "format.accus_epochs", labelKey: "accus_epochs", type: "number" });
      html += fieldHtml({ path: "format.batch_size", labelKey: "batch_size", type: "number" });
      html += fieldHtml({ path: "format.network_train_unet_only", labelKey: "network_train_unet_only", type: "checkbox" });
      html += renderArgList("format.network_args", "network_args");
      break;
    case "saving":
      html += fieldHtml({ path: "saving.output_dir", labelKey: "output_dir", full: true });
      html += fieldHtml({ path: "saving.logging_dir", labelKey: "logging_dir", full: true });
      html += fieldHtml({ path: "saving.output_name", labelKey: "output_name" });
      html += fieldHtml({ path: "saving.save_model_as", labelKey: "save_model_as" });
      html += fieldHtml({ path: "saving.save_every_n_steps", labelKey: "save_every_n_steps", type: "number" });
      html += fieldHtml({ path: "saving.save_every_n_epochs", labelKey: "save_every_n_epochs", type: "number" });
      html += fieldHtml({ path: "saving.save_last_n_steps", labelKey: "save_last_n_steps", type: "number" });
      html += fieldHtml({ path: "saving.save_last_n_epochs", labelKey: "save_last_n_epochs", type: "number" });
      break;
    case "resume":
      html += fieldHtml({ path: "resume.enabled", labelKey: "resume_enabled", type: "checkbox" });
      html += fieldHtml({ path: "resume.path", labelKey: "resume_path", full: true });
      break;
    case "sample":
      html += fieldHtml({ path: "sample.enabled", labelKey: "sample_enabled", type: "checkbox" });
      html += fieldHtml({ path: "sample.sample_every_n_steps", labelKey: "sample_every_n_steps", type: "number" });
      html += fieldHtml({ path: "sample.sample_every_n_epochs", labelKey: "sample_every_n_epochs", type: "number" });
      html += fieldHtml({ path: "sample.sample_sampler", labelKey: "sample_sampler", type: "select", choices: samplerChoices });
      html += fieldHtml({ path: "sample.sample_steps", labelKey: "sample_steps", type: "number" });
      html += fieldHtml({ path: "sample.cfg_scale", labelKey: "cfg_scale", type: "number" });
      html += fieldHtml({ path: "sample.width", labelKey: "width", type: "number" });
      html += fieldHtml({ path: "sample.height", labelKey: "height", type: "number" });
      html += fieldHtml({ path: "sample.seed", labelKey: "seed", type: "number" });
      html += fieldHtml({ path: "sample.prompt", labelKey: "prompt", type: "textarea", full: true });
      html += fieldHtml({ path: "sample.negative_prompt", labelKey: "negative_prompt", type: "textarea", full: true });
      break;
    case "execute":
      html += renderArgList("raw_args", "raw_args");
      html += fieldHtml({ path: "execute.tb_logdir", labelKey: "tb_logdir", full: true });
      html += fieldHtml({ path: "execute.tb_port", labelKey: "tb_port", type: "number" });
      break;
    default:
      html = "";
  }

  el.sectionHelp.textContent = t(`help.${state.activeSection}`);
  el.sectionContent.innerHTML = html;
}

function renderProfileTabs() {
  el.profileTabs.innerHTML = "";
  state.profiles.forEach((profile) => {
    const btn = document.createElement("button");
    btn.textContent = profile.name;
    btn.className = profile.id === state.activeProfileId ? "active" : "";
    btn.addEventListener("click", () => {
      state.activeProfileId = profile.id;
      schedulePersist();
      render();
    });
    el.profileTabs.appendChild(btn);
  });

  const addBtn = document.createElement("button");
  addBtn.textContent = "+";
  addBtn.addEventListener("click", onAddProfile);
  el.profileTabs.appendChild(addBtn);
}

function renderStepNav() {
  el.stepNav.innerHTML = "";
  const profile = currentProfile();
  for (const step of state.schema.steps || DEFAULT_SCHEMA.steps) {
    const s = profile ? computeStepState(profile, step.id) : "warning";
    const btn = document.createElement("button");
    btn.className = `step-btn ${step.id === state.activeStep ? "active" : ""}`;
    btn.setAttribute("data-step", step.id);
    btn.innerHTML = `<span class="step-label">${escapeHtml(t(`tabs.${step.id}`))}</span><span class="step-badge ${s}">${escapeHtml(t(`step_state.${s}`))}</span>`;
    el.stepNav.appendChild(btn);
  }
}

function renderSectionTabs() {
  const sections = sectionsForStep(state.activeStep);
  if (!sections.includes(state.activeSection)) state.activeSection = sections[0];

  el.sectionTabs.innerHTML = "";
  sections.forEach((key) => {
    const btn = document.createElement("button");
    btn.textContent = t(`tabs.${key}`);
    btn.className = key === state.activeSection ? "active" : "";
    btn.setAttribute("data-section", key);
    el.sectionTabs.appendChild(btn);
  });
}

function renderMessage() {
  el.messages.innerHTML = state.lastMessage ? `<div class="${state.lastMessageClass}">${escapeHtml(state.lastMessage)}</div>` : "";
}

function renderActionButtons() {
  const l = state.loading;
  el.actionValidate.textContent = l.validate ? t("messages.loading") : t("fields.validate");
  el.actionStart.textContent = l.start ? t("messages.loading") : t("fields.start_batch");
  el.actionStop.textContent = l.stop ? t("messages.loading") : t("fields.stop_batch");
  el.actionStartTb.textContent = l.startTb ? t("messages.loading") : t("fields.start_tb");
  el.actionStopTb.textContent = l.stopTb ? t("messages.loading") : t("fields.stop_tb");

  el.actionValidate.disabled = l.validate || l.start;
  el.actionStart.disabled = l.start || l.validate;
  el.actionStop.disabled = l.stop;
  el.actionStartTb.disabled = l.startTb;
  el.actionStopTb.disabled = l.stopTb;
}

function renderStatusCards() {
  const summary = state.summary || {
    batch_running: false,
    total_jobs: 0,
    queued: 0,
    running: 0,
    failed: 0,
    succeeded: 0,
    cancelled: 0,
    queue_remaining: 0,
    current_job: null,
    last_error: null,
    tensorboard: { running: false }
  };
  const status = state.status || { jobs: [], tensorboard: {} };

  let pillClass = "status-pill idle";
  let batchStateText = t("status.idle");
  if (summary.batch_running) {
    pillClass = "status-pill running";
    batchStateText = t("status.running");
  } else if (summary.failed > 0) {
    pillClass = "status-pill error";
    batchStateText = t("status.finished_with_error");
  } else if (summary.succeeded > 0 && summary.queued === 0 && summary.running === 0) {
    pillClass = "status-pill ok";
    batchStateText = t("status.finished");
  }

  el.cardBatchState.className = pillClass;
  el.cardBatchState.textContent = batchStateText;
  el.cardCurrentJob.textContent = summary.current_job?.name || "-";
  el.cardQueueCount.textContent = `${summary.queue_remaining}`;
  el.cardTotalCount.textContent = `${summary.total_jobs}`;
  el.cardLastError.textContent = summary.last_error?.message || "-";
  el.cardTbStatus.textContent = summary.tensorboard?.running ? `${t("status.running")} (${summary.tensorboard.url || ""})` : t("status.idle");

  const jobs = Array.isArray(status.jobs) ? status.jobs : [];
  el.statusJobs.textContent = jobs.map((job) => `${job.name} [${job.status}] ${job.message || ""}`).join("\n") || "-";
  const runningJob = jobs.find((j) => j.id === status.current_job_id) || jobs.find((j) => j.status === "running") || null;
  el.logPreview.textContent = runningJob?.log_tail || status?.tensorboard?.log_tail || "";
}

function render() {
  if (!currentProfile() && state.profiles.length > 0) state.activeProfileId = state.profiles[0].id;

  el.appTitle.textContent = t("app_title");
  document.title = t("app_title");
  document.documentElement.lang = isZhLocale() ? "zh-Hant" : "en";
  el.themeLabel.textContent = t("theme_label");
  el.themeSelect.value = state.theme_preference;
  Array.from(el.themeSelect.options).forEach((option) => {
    if (option.value === "system") option.textContent = t("theme_system");
    if (option.value === "light") option.textContent = t("theme_light");
    if (option.value === "dark") option.textContent = t("theme_dark");
  });
  el.langLabel.textContent = t("lang_label");
  el.addProfileBtn.textContent = t("add_profile");
  el.duplicateProfileBtn.textContent = t("duplicate_profile");
  el.deleteProfileBtn.textContent = t("delete_profile");
  el.exportBtn.textContent = t("export_json");
  el.importBtn.textContent = t("import_json");
  el.stepsTitle.textContent = t("steps_title");
  el.labelProfileName.textContent = t("fields.profile_name");
  el.labelRunEnabled.textContent = t("run_enabled");

  el.statusTitle.textContent = t("fields.batch_status");
  el.cardBatchStateLabel.textContent = t("status_cards.batch_state");
  el.cardCurrentJobLabel.textContent = t("status_cards.current_job");
  el.cardQueueLabel.textContent = t("status_cards.queue");
  el.cardTotalLabel.textContent = t("status_cards.total");
  el.cardLastErrorLabel.textContent = t("status_cards.last_error");
  el.cardTbLabel.textContent = t("status_cards.tensorboard");
  el.jobsTitle.textContent = t("status_cards.jobs");
  el.logPreviewTitle.textContent = t("fields.log_preview");
  el.langSelect.setAttribute("aria-label", t("aria.language_select", "Language"));
  el.themeSelect.setAttribute("aria-label", t("aria.theme_select", "Theme"));
  el.stepNav.setAttribute("aria-label", t("aria.step_nav", "Workflow Steps"));
  el.profileNameInput.setAttribute("aria-label", t("aria.profile_name", t("fields.profile_name")));
  el.runEnabledInput.setAttribute("aria-label", t("aria.run_enabled", t("run_enabled")));

  if (state.defaultsWarning) {
    el.defaultsWarning.hidden = false;
    el.defaultsWarning.textContent = state.defaultsWarning;
  } else {
    el.defaultsWarning.hidden = true;
    el.defaultsWarning.textContent = "";
  }

  renderProfileTabs();
  renderStepNav();
  renderSectionTabs();

  const active = currentProfile();
  if (active) {
    el.profileNameInput.value = active.name || "";
    el.runEnabledInput.checked = !!active.run_enabled;
    renderSection();
  }

  renderActionButtons();
  renderStatusCards();
  renderMessage();
}

function setMessage(text, isError = false) {
  state.lastMessage = text;
  state.lastMessageClass = isError ? "error" : "ok";
  renderMessage();
}
function schedulePersist() {
  if (state.saveTimer) clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => {
    localStorage.setItem(
      STORAGE_KEY_V2,
      JSON.stringify({
        version: 2,
        lang: state.lang,
        layout_mode: state.layout_mode,
        theme_preference: state.theme_preference,
        active_profile_id: state.activeProfileId,
        active_step: state.activeStep,
        profiles: state.profiles,
        recent_paths: state.recentPaths
      })
    );
  }, 200);
}

function restoreFromStorage() {
  let parsed = null;
  const current = localStorage.getItem(STORAGE_KEY_V2);
  if (current) {
    try {
      parsed = JSON.parse(current);
    } catch {
      parsed = null;
    }
  }

  if (!parsed) {
    for (const key of STORAGE_KEYS_LEGACY) {
      const legacy = localStorage.getItem(key);
      if (!legacy) continue;
      try {
        parsed = JSON.parse(legacy);
        break;
      } catch {
        parsed = null;
      }
    }
  }

  if (!parsed) {
    const p = createDefaultProfile(1);
    state.profiles = [p];
    state.activeProfileId = p.id;
    state.activeStep = "model";
    state.activeSection = "model";
    state.theme_preference = "system";
    state.recentPaths = [];
    return;
  }

  const normalized = normalizeState(parsed);
  state.lang = normalized.lang || state.lang;
  state.layout_mode = normalized.layout_mode || "dashboard";
  state.theme_preference = normalizeThemePreference(normalized.theme_preference);
  state.profiles = normalized.profiles;
  state.activeProfileId = normalized.active_profile_id;
  state.activeStep = normalized.active_step;
  state.activeSection = sectionsForStep(state.activeStep)[0];
  state.recentPaths = normalized.recent_paths;
}

function addRecentPath(path) {
  if (typeof path !== "string") return;
  const normalized = path.trim();
  if (!normalized) return;
  state.recentPaths = [normalized, ...state.recentPaths.filter((x) => x !== normalized)].slice(0, 12);
}

async function api(url, options = {}) {
  const res = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.error || `Request failed: ${res.status}`);
  return data;
}

async function withLoading(key, fn) {
  if (state.loading[key]) return;
  state.loading[key] = true;
  renderActionButtons();
  try {
    await fn();
  } finally {
    state.loading[key] = false;
    renderActionButtons();
  }
}

function onAddProfile() {
  const profile = createDefaultProfile(state.profiles.length + 1);
  state.profiles.push(profile);
  state.activeProfileId = profile.id;
  state.activeStep = "model";
  state.activeSection = "model";
  schedulePersist();
  render();
}

function onDuplicateProfile() {
  const profile = currentProfile();
  if (!profile) return;
  const copy = deepMerge(createDefaultProfile(state.profiles.length + 1), JSON.parse(JSON.stringify(profile)));
  copy.id = uuid();
  copy.name = `${profile.name} ${taskCopySuffix()}`;
  copy.meta.updated_at = Date.now() / 1000;
  state.profiles.push(copy);
  state.activeProfileId = copy.id;
  schedulePersist();
  render();
}

function onDeleteProfile() {
  if (state.profiles.length <= 1) return;
  state.profiles = state.profiles.filter((p) => p.id !== state.activeProfileId);
  state.activeProfileId = state.profiles[0]?.id || null;
  schedulePersist();
  render();
}

async function onValidateProfile() {
  const profile = currentProfile();
  if (!profile) return;

  await withLoading("validate", async () => {
    const result = await api("/api/profile/validate", { method: "POST", body: JSON.stringify({ profile }) });

    profile.execute.validation = {
      errors: result.errors,
      warnings: result.warnings,
      field_errors: result.field_errors,
      field_warnings: result.field_warnings,
      valid: result.valid
    };
    profile.meta.validation_state = profile.execute.validation;
    profile.meta.updated_at = Date.now() / 1000;

    if (!result.valid) {
      const firstPath = Object.keys(result.field_errors || {})[0];
      if (firstPath) {
        state.activeStep = stepForPath(firstPath);
        state.activeSection = sectionsForStep(state.activeStep)[0];
      }
    }

    schedulePersist();
    setMessage(result.valid ? t("messages.validation_ok") : t("messages.validation_fail"), !result.valid);
    render();
  });
}

async function onStartBatch() {
  await withLoading("start", async () => {
    await api("/api/batch/start", { method: "POST", body: JSON.stringify({ profiles: state.profiles }) });
    setMessage(t("messages.batch_started"));
    await refreshStatus();
  });
}

async function onStopBatch() {
  await withLoading("stop", async () => {
    await api("/api/batch/stop", { method: "POST", body: "{}" });
    setMessage(t("messages.batch_stopped"));
    await refreshStatus();
  });
}

async function onInspectDataset() {
  const profile = currentProfile();
  if (!profile) return;
  try {
    const result = await api("/api/dataset/inspect", { method: "POST", body: JSON.stringify({ path: profile.dataset.dataset_path || "" }) });
    profile.dataset.inspect_result = result.result;
    profile.meta.updated_at = Date.now() / 1000;
    addRecentPath(profile.dataset.dataset_path || "");
    schedulePersist();
    render();
  } catch (err) {
    setMessage(String(err), true);
  }
}

async function onStartTensorBoard() {
  const profile = currentProfile();
  if (!profile) return;
  const logdir = profile.execute.tb_logdir || profile.saving.logging_dir || profile.saving.output_dir;
  const port = Number(profile.execute.tb_port || 6006);

  await withLoading("startTb", async () => {
    await api("/api/tensorboard/start", { method: "POST", body: JSON.stringify({ logdir, port }) });
    setMessage(t("messages.tb_started"));
    await refreshStatus();
  });
}

async function onStopTensorBoard() {
  await withLoading("stopTb", async () => {
    await api("/api/tensorboard/stop", { method: "POST", body: "{}" });
    setMessage(t("messages.tb_stopped"));
    await refreshStatus();
  });
}

function exportState() {
  const payload = {
    version: 2,
    ui_state: {
      version: 2,
      lang: state.lang,
      layout_mode: state.layout_mode,
      theme_preference: state.theme_preference,
      active_profile_id: state.activeProfileId,
      active_step: state.activeStep,
      profiles: state.profiles,
      recent_paths: state.recentPaths
    }
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "anima_ui_profiles.json";
  a.click();
  URL.revokeObjectURL(url);
}

async function importState(file) {
  if (!file) return;
  await withLoading("import", async () => {
    let parsed;
    try {
      parsed = JSON.parse(await file.text());
    } catch {
      throw new Error(t("messages.import_parse_error"));
    }

    const result = await api("/api/profile/import", { method: "POST", body: JSON.stringify(parsed) });
    const normalized = normalizeState(result.state);
    const importedThemeRaw = parsed?.ui_state?.theme_preference ?? parsed?.theme_preference;
    const importedTheme = normalizeThemePreference(importedThemeRaw);

    state.lang = normalized.lang;
    state.theme_preference = importedThemeRaw ? importedTheme : normalizeThemePreference(normalized.theme_preference);
    applyTheme();
    state.profiles = normalized.profiles;
    state.activeProfileId = normalized.active_profile_id;
    state.activeStep = normalized.active_step;
    state.activeSection = sectionsForStep(state.activeStep)[0];
    state.recentPaths = normalized.recent_paths;

    schedulePersist();
    setMessage(result.warnings?.length ? `${t("messages.import_ok")} (${result.warnings.join(", ")})` : t("messages.import_ok"));
    await loadI18n();
    render();
  });
}

async function refreshStatus() {
  try {
    const [statusPayload, summaryPayload] = await Promise.all([api("/api/batch/status"), api("/api/batch/summary")]);
    state.status = statusPayload;
    state.summary = summaryPayload.summary;
    renderStatusCards();
  } catch {
    // keep last status
  }
}

function stopPollingFallback() {
  if (state.pollingTimer) {
    clearInterval(state.pollingTimer);
    state.pollingTimer = null;
  }
}

function startPollingFallback() {
  if (state.pollingTimer) return;
  state.pollingTimer = setInterval(refreshStatus, 2000);
}

function connectBatchEvents() {
  if (state.eventSource) state.eventSource.close();
  try {
    const source = new EventSource("/api/batch/events");
    state.eventSource = source;

    source.addEventListener("snapshot", (evt) => {
      try {
        const payload = JSON.parse(evt.data);
        state.status = payload.status || state.status;
        state.summary = payload.summary || state.summary;
        renderStatusCards();
      } catch {
        // ignore malformed event payload
      }
    });

    source.onopen = () => stopPollingFallback();
    source.onerror = () => {
      if (state.eventSource) {
        state.eventSource.close();
        state.eventSource = null;
      }
      startPollingFallback();
    };
  } catch {
    startPollingFallback();
  }
}

function bindEvents() {
  el.themeSelect.addEventListener("change", (e) => {
    state.theme_preference = normalizeThemePreference(e.target.value);
    applyTheme();
    schedulePersist();
    render();
  });

  el.langSelect.addEventListener("change", async (e) => {
    state.lang = e.target.value;
    await loadI18n();
    schedulePersist();
    render();
  });

  el.addProfileBtn.addEventListener("click", onAddProfile);
  el.duplicateProfileBtn.addEventListener("click", onDuplicateProfile);
  el.deleteProfileBtn.addEventListener("click", onDeleteProfile);
  el.exportBtn.addEventListener("click", exportState);
  el.importBtn.addEventListener("click", () => el.importFile.click());

  el.importFile.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      await importState(file);
    } catch (err) {
      setMessage(String(err), true);
    }
  });

  el.profileNameInput.addEventListener("input", (e) => {
    const profile = currentProfile();
    if (!profile) return;
    profile.name = e.target.value;
    profile.meta.updated_at = Date.now() / 1000;
    schedulePersist();
    renderProfileTabs();
  });

  el.runEnabledInput.addEventListener("change", (e) => {
    const profile = currentProfile();
    if (!profile) return;
    profile.run_enabled = e.target.checked;
    profile.meta.updated_at = Date.now() / 1000;
    schedulePersist();
  });

  el.stepNav.addEventListener("click", (e) => {
    const target = e.target.closest("[data-step]");
    if (!target) return;
    state.activeStep = target.dataset.step;
    state.activeSection = sectionsForStep(state.activeStep)[0];
    schedulePersist();
    render();
  });

  el.sectionTabs.addEventListener("click", (e) => {
    const target = e.target.closest("[data-section]");
    if (!target) return;
    state.activeSection = target.dataset.section;
    schedulePersist();
    renderSection();
  });

  const onContentEdit = (target) => {
    const profile = currentProfile();
    if (!profile) return;

    if (target.dataset.path) {
      const path = target.dataset.path;
      const value = target.type === "checkbox" ? target.checked : target.type === "number" ? (target.value === "" ? "" : Number(target.value)) : target.value;
      setByPath(profile, path, value);
      profile.meta.updated_at = Date.now() / 1000;
      if (path.endsWith("_path") && typeof value === "string") addRecentPath(value);
      if (path.startsWith("precision_opt.cache_") || path.startsWith("dataset.")) render();
      schedulePersist();
    }

    if (target.dataset.argPath) {
      const listPath = target.dataset.argPath;
      const idx = Number(target.dataset.argIndex);
      const list = getByPath(profile, listPath, []);
      list[idx] = target.value;
      setByPath(profile, listPath, list);
      profile.meta.updated_at = Date.now() / 1000;
      schedulePersist();
    }
  };

  el.sectionContent.addEventListener("input", (e) => onContentEdit(e.target));
  el.sectionContent.addEventListener("change", (e) => onContentEdit(e.target));

  el.sectionContent.addEventListener("click", (e) => {
    const target = e.target;
    const profile = currentProfile();
    if (!profile) return;

    if (target.dataset.argAdd) {
      const listPath = target.dataset.argAdd;
      const list = getByPath(profile, listPath, []);
      list.push("");
      setByPath(profile, listPath, list);
      profile.meta.updated_at = Date.now() / 1000;
      schedulePersist();
      renderSection();
      return;
    }

    if (target.dataset.argRemove) {
      const listPath = target.dataset.argRemove;
      const idx = Number(target.dataset.argIndex);
      const list = getByPath(profile, listPath, []);
      list.splice(idx, 1);
      setByPath(profile, listPath, list);
      profile.meta.updated_at = Date.now() / 1000;
      schedulePersist();
      renderSection();
      return;
    }

    if (target.id === "btn-inspect-dataset") onInspectDataset();
  });

  el.actionValidate.addEventListener("click", onValidateProfile);
  el.actionStart.addEventListener("click", onStartBatch);
  el.actionStop.addEventListener("click", onStopBatch);
  el.actionStartTb.addEventListener("click", onStartTensorBoard);
  el.actionStopTb.addEventListener("click", onStopTensorBoard);
}

async function loadI18n() {
  const res = await fetch(`/i18n/${state.lang}.json`);
  state.i18n = await res.json();
  el.langSelect.value = state.lang;
}

async function loadDefaults() {
  const payload = await api("/api/defaults");
  state.defaults = payload.defaults || { fields: {} };
  state.defaultsWarning = payload.warning || "";
}

async function loadSchema() {
  try {
    const payload = await api("/api/profile/schema");
    state.schema = payload.schema || DEFAULT_SCHEMA;
  } catch {
    state.schema = DEFAULT_SCHEMA;
  }
}

async function init() {
  bindEvents();
  await loadDefaults();
  restoreFromStorage();
  applyTheme();
  bindThemeEvents();
  await loadI18n();
  await loadSchema();

  const activeSections = sectionsForStep(state.activeStep);
  if (!activeSections.includes(state.activeSection)) state.activeSection = activeSections[0];

  render();
  await refreshStatus();
  connectBatchEvents();
  startPollingFallback();
}

init();
