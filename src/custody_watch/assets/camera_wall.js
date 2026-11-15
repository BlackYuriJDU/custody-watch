"use strict";

// A seleção é local ao andar; mudar o layout não muda câmera/segmento.
function cameraWallLayout(cameras, floorId, selected, mode) {
  if (mode !== "mosaic" && mode !== "focus") throw new RangeError("Layout de câmeras inválido.");
  const visible = cameras.filter(camera => camera.floor_id === floorId);
  const active = visible.find(camera => camera.camera_id === selected) || visible[0];
  return {
    selected: active?.camera_id || null,
    cameras: mode === "focus" && active
      ? [active, ...visible.filter(camera => camera !== active)] : visible,
  };
}

if (typeof module !== "undefined") module.exports = {cameraWallLayout};
