"use strict";

class ReviewQueueUI {
  constructor(config, reviewer, openItem, onClear) {
    this.$ = id => document.getElementById(id);
    this.config = config; this.reviewer = reviewer; this.openItem = openItem; this.onClear = onClear;
    this.cursor = null; this.busy = false; this.generation = 0; this.focused = false; this.imageState = "idle";
    this.$("queue-file").addEventListener("change", event => { void this.load(event.target.files[0]); });
    for (const suffix of ["", "-modal"]) {
      this.$(`queue-prev${suffix}`).addEventListener("click", () => { void this.navigate(-1); });
      this.$(`queue-next${suffix}`).addEventListener("click", () => { void this.navigate(1); });
      this.$(`queue-open${suffix}`).addEventListener("click", () => { void this.navigate(0); });
    }
    this.$("queue-clear").addEventListener("click", () => this.clear());
    this.render();
  }
  feedback(message) {
    for (const id of ["queue-message", "queue-message-modal"]) this.$(id).textContent = message;
  }
  render() {
    const cursor = this.cursor, item = cursor?.current;
    const progress = cursor?.progress(this.reviewer.store);
    const position = item ? `Quadro ${cursor.index+1}/${cursor.queue.items.length} · ${item.camera_id} · índice ${item.frame_index} · ${item.global_t.toLocaleString("pt-BR")} s · ${item.local_track_ids.length} IDs` : "Nenhuma fila carregada.";
    this.$("queue-position").textContent = position;
    this.$("queue-position-modal").textContent = this.focused ? position : `Inspeção livre. ${position}`;
    const counts = progress ? `${progress.frames_done}/${progress.frames_total} quadros completos · ${progress.observations_done}/${progress.observations_total} anotações guardadas nesta página` : "";
    for (const id of ["queue-progress", "queue-progress-modal"]) this.$(id).textContent = counts;
    this.$("queue-navigation-modal").hidden = !cursor;
    this.$("queue-file").disabled = this.busy;
    this.$("queue-clear").disabled = !cursor && !this.busy;
    for (const suffix of ["", "-modal"]) {
      this.$(`queue-open${suffix}`).disabled = !cursor || this.busy;
      this.$(`queue-prev${suffix}`).disabled = !cursor || this.busy || cursor.index === 0;
      this.$(`queue-next${suffix}`).disabled = !cursor || this.busy || cursor.index === cursor.queue.items.length-1;
    }
    if (!item) this.$("queue-policy").textContent = "";
    else {
      const policy = cursor.queue.selection;
      const origin = item.prediction_provenance === "ground_truth" ? "ORACLE / GABARITO" : "PREDIÇÕES DO DETECTOR";
      this.$("queue-policy").textContent = `${origin} · seleção declarada ${policy.algorithm} · seed ${policy.seed} · até ${policy.frames_per_camera} quadros/câmera · intervalo ${policy.min_gap_s} s · ${cursor.queue.excluded_cameras.length} câmeras excluídas. Amostra de conveniência; não valida confiança.`;
    }
  }
  async load(file) {
    if (!file) return;
    const ticket = ++this.generation;
    this.busy = true; this.focused = false; this.render();
    try {
      if (file.size > ReviewQueue.MAX_BYTES) throw new Error("Arquivo excede o limite de 1 MiB.");
      const buffer = await file.arrayBuffer();
      if (ticket !== this.generation) return;
      const text = new TextDecoder("utf-8", {fatal:true, ignoreBOM:true}).decode(buffer);
      this.cursor = new ReviewQueue.QueueCursor(ReviewQueue.parseQueue(text,this.config));
      this.feedback("Fila pronta. Abra o quadro selecionado. Nenhuma anotação foi criada.");
    } catch (error) {
      if (ticket === this.generation) {
        this.$("queue-file").value = "";
        this.feedback(`Fila recusada: ${error instanceof TypeError ? "JSON deve usar UTF-8 válido." : error.message} O rascunho e a fila anterior foram preservados.`);
      }
    } finally {
      if (ticket === this.generation) { this.busy = false; this.render(); }
    }
  }
  clear() {
    ++this.generation; this.cursor = null; this.busy = false; this.focused = false;
    this.$("queue-file").value = ""; this.onClear();
    this.feedback("Fila removida. As anotações guardadas permanecem no rascunho."); this.render();
  }
  async navigate(delta) {
    if (!this.cursor || this.busy) return;
    if (delta && !this.cursor.move(delta)) return;
    const ticket = ++this.generation, item = this.cursor.current;
    this.busy = true; this.focused = false; this.imageState = "waiting";
    this.feedback("Buscando e conferindo o quadro da fila…"); this.render();
    try {
      const opened = await this.openItem(item, () => ticket === this.generation);
      if (ticket === this.generation) {
        this.focused = opened;
        if (!opened) this.feedback("Busca cancelada ou não atualizada. Revisão da fila não foi aberta.");
        else if (this.imageState === "waiting") this.feedback("Quadro vinculado. Aguarde o carregamento da imagem para anotar.");
      }
    } catch (error) {
      if (ticket === this.generation) this.feedback(error.message);
    } finally {
      if (ticket === this.generation) { this.busy = false; this.render(); }
    }
  }
  inspectionOpened(item) {
    this.focused = Boolean(item && this.cursor?.current === item);
    this.render();
  }
  imageReady(item) {
    if (item && this.cursor?.current === item) {
      this.imageState = "ready";
      this.feedback("Quadro e IDs conferidos. Escolha um ID e declare estado, origem e método.");
    }
  }
  imageFailed(item) {
    if (item && this.cursor?.current === item) {
      this.imageState = "failed";
      this.feedback("Imagem indisponível; anotação bloqueada. O rascunho permanece intacto.");
    }
  }
}
