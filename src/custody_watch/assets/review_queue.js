"use strict";

const ReviewQueue = (() => {
  const review = typeof module !== "undefined" && module.exports ? require("./contact_review.js") : ContactReview;
  const MAX_BYTES = 1024 * 1024;
  const fail = message => { throw new Error(message); };
  const integer = value => Number.isSafeInteger(value) && value >= 0;
  const finite = value => typeof value === "number" && Number.isFinite(value);
  function fields(value, expected) {
    if (!value || typeof value !== "object" || Array.isArray(value)) fail("Objeto inválido na fila.");
    const actual = Object.keys(value);
    if (actual.length !== expected.length || actual.some(key => !expected.includes(key))) fail("Campos incompatíveis na fila.");
  }
  function strictJSON(text) {
    if (typeof text !== "string" || text.length > MAX_BYTES || new TextEncoder().encode(text).length > MAX_BYTES) fail("Arquivo excede o limite de 1 MiB.");
    const stack = [];
    try {
      for (let position=0;position<text.length;position++) {
        const token = text[position];
        if (token === '"') {
          const start = position++;
          let closed = false;
          for (;position<text.length;position++) {
            if (text[position] === "\\") position++;
            else if (text[position] === '"') { closed = true; break; }
          }
          if (!closed) fail("Arquivo JSON inválido.");
          const top = stack.at(-1);
          if (top?.object && top.key) {
            const key = JSON.parse(text.slice(start,position+1));
            if (top.seen.has(key)) fail("Chave JSON duplicada.");
            top.seen.add(key); top.key = false;
          }
        }
        else if (token === "{" || token === "[") {
          stack.push({object:token === "{", key:true, seen:new Set()});
          if (stack.length > 64) fail("JSON excede o limite de profundidade.");
        } else if (token === "}" || token === "]") stack.pop();
        else {
          const top = stack.at(-1);
          if (token === "," && top?.object) top.key = true;
        }
      }
      return JSON.parse(text);
    } catch (error) {
      if (error instanceof SyntaxError) fail("Arquivo JSON inválido.");
      throw error;
    }
  }
  function selection(value) {
    fields(value,["algorithm","seed","frames_per_camera","min_gap_s","unit"]);
    if (value.algorithm !== "sha256_rank_greedy_gap_v1" || value.unit !== "camera_frame") fail("Seleção da fila não suportada.");
    if (!integer(value.seed) || value.seed > 2**32-1 || !integer(value.frames_per_camera) || value.frames_per_camera < 1 || value.frames_per_camera > 1000 || !finite(value.min_gap_s) || value.min_gap_s < 0) fail("Parâmetros inválidos na fila.");
    return Object.freeze({...value});
  }
  function parseQueue(text, config) {
    const data = strictJSON(text);
    fields(data,["schema_version","kind","registry_sha256","selection","cameras","excluded_cameras"]);
    if (data.schema_version !== 1 || data.kind !== "ground_contact_review_queue") fail("Arquivo não é uma fila de revisão v1.");
    if (!/^[a-f0-9]{64}$/.test(data.registry_sha256) || data.registry_sha256 !== config.registry_sha256) fail("Fila pertence a outro cadastro.");
    const policy = selection(data.selection), items = [], seen = new Set();
    if (!Array.isArray(data.cameras) || !Array.isArray(data.excluded_cameras) || data.cameras.length + data.excluded_cameras.length > 256) fail("Câmeras inválidas ou limite excedido.");
    let observations = 0;
    for (const camera of data.cameras) {
      fields(camera,["camera_id","segment_id","sequence_sha256","prediction_provenance","candidate_frames","selected_frames","frames"]);
      const known = config.cameras.find(item => item.camera_id === camera.camera_id);
      if (!known || seen.has(camera.camera_id)) fail("Câmera desconhecida ou duplicada.");
      seen.add(camera.camera_id);
      if (camera.segment_id !== known.segment_id || camera.sequence_sha256 !== known.sequence_sha256 || camera.prediction_provenance !== known.provenance || known.source_kind !== "video") fail("Segmento, sequência ou origem divergente.");
      if (!integer(camera.candidate_frames) || !integer(camera.selected_frames) || !Array.isArray(camera.frames) || camera.selected_frames !== camera.frames.length || camera.selected_frames > camera.candidate_frames || camera.selected_frames > policy.frames_per_camera) fail("Contagem de quadros divergente.");
      let previous = null;
      for (const frame of camera.frames) {
        fields(frame,["frame_index","media_t","global_t","local_track_ids"]);
        if (!integer(frame.frame_index) || !finite(frame.media_t) || !finite(frame.global_t) || !finite(known.clock?.offset_s) || frame.global_t !== frame.media_t + known.clock.offset_s || frame.global_t < config.timeline.start_s || frame.global_t >= config.timeline.end_s) fail("Quadro ou tempo incompatível com o relógio.");
        if (previous && (frame.frame_index <= previous.frame_index || frame.global_t <= previous.global_t || frame.global_t - previous.global_t < policy.min_gap_s)) fail("Quadro duplicado, fora de ordem ou intervalo inválido.");
        if (!Array.isArray(frame.local_track_ids) || frame.local_track_ids.length === 0) fail("Quadro sem IDs para revisão.");
        for (let i=0;i<frame.local_track_ids.length;i++) {
          const id = frame.local_track_ids[i];
          if (!integer(id) || (i > 0 && id <= frame.local_track_ids[i-1])) fail("ID inválido, duplicado ou fora de ordem.");
        }
        observations += frame.local_track_ids.length;
        if (observations > 1000 || items.length >= 1000) fail("Fila excede o limite de 1000 quadros/IDs.");
        items.push(Object.freeze({
          registry_sha256:data.registry_sha256, camera_id:camera.camera_id, segment_id:camera.segment_id,
          sequence_sha256:camera.sequence_sha256, prediction_provenance:camera.prediction_provenance,
          frame_index:frame.frame_index, media_t:frame.media_t, global_t:frame.global_t,
          local_track_ids:Object.freeze([...frame.local_track_ids]),
        }));
        previous = frame;
      }
    }
    const excluded = data.excluded_cameras.map(camera => {
      fields(camera,["camera_id","reason"]);
      if (!config.cameras.some(item => item.camera_id === camera.camera_id) || seen.has(camera.camera_id) || !["schematic","unavailable"].includes(camera.reason)) fail("Exclusão inválida ou duplicada.");
      seen.add(camera.camera_id);
      return Object.freeze({...camera});
    });
    if (seen.size !== config.cameras.length) fail("Câmera do cadastro omitida na fila.");
    if (items.length === 0) fail("Fila vazia.");
    return Object.freeze({registry_sha256:data.registry_sha256,selection:policy,items:Object.freeze(items),excluded_cameras:Object.freeze(excluded)});
  }
  function matchFrame(item, config, camera, snapshot) {
    const context = review.createReviewContext(config,camera,snapshot);
    if (!context || snapshot.status !== "observed" || item.registry_sha256 !== config.registry_sha256 || context.camera_id !== item.camera_id || context.segment_id !== item.segment_id || context.sequence_sha256 !== item.sequence_sha256 || context.provenance !== item.prediction_provenance || context.frame_index !== item.frame_index || context.t !== item.global_t || context.media_t !== item.media_t) fail("Quadro da fila indisponível ou divergente; revisão bloqueada.");
    const tracks = item.local_track_ids.map(id => context.tracks.find(track => track.local_track_id === id && track.cls === "person"));
    if (tracks.some(track => !track)) fail("ID da fila ausente nesta imagem; revisão bloqueada.");
    return Object.freeze({...context,tracks:Object.freeze(tracks)});
  }
  class QueueCursor {
    #index = 0;
    constructor(queue) { this.queue = queue; Object.freeze(this); }
    get index() { return this.#index; }
    get current() { return this.queue.items[this.#index]; }
    move(delta) {
      if (![-1,1].includes(delta)) fail("Passo de navegação inválido.");
      const next = this.#index + delta;
      if (next < 0 || next >= this.queue.items.length) return false;
      this.#index = next; return true;
    }
    progress(store) {
      let done = 0, observations = 0, complete = 0;
      for (const item of this.queue.items) {
        const count = item.local_track_ids.filter(id => store?.get(item,id)).length;
        done += count; observations += item.local_track_ids.length;
        if (count === item.local_track_ids.length) complete++;
      }
      return {frames_total:this.queue.items.length,frames_done:complete,observations_total:observations,observations_done:done};
    }
  }
  return {parseQueue,matchFrame,QueueCursor,MAX_BYTES};
})();
if (typeof module !== "undefined") module.exports = ReviewQueue;
