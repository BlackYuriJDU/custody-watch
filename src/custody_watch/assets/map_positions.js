"use strict";

// O cache v1 não contém evidência de contato com o chão. Nunca promover a
// base de uma caixa (incluindo oracle) a localização para associação.
function mapPosition(track, showDiagnostics = false) {
  const statuses = ["ground_contact_unknown", "image_border"];
  const point = track.projected_position_m;
  if (showDiagnostics !== true || !statuses.includes(track.position_status)) return null;
  if (!Array.isArray(point) || point.length !== 2 || !point.every(Number.isFinite)) return null;
  return {point, diagnostic: true};
}

// Cores da planta vêm das custom properties do tema (multicamera.css), nunca
// de literais. `read(nome)` devolve o valor bruto da propriedade, como
// `getComputedStyle(raiz).getPropertyValue`; vazio cai em currentColor.
const MAP_CAMERA_TOKENS = ["--camera-0", "--camera-1", "--camera-2", "--camera-3"];

function mapTheme(read) {
  const color = name => String(read(name) || "").trim() || "currentColor";
  return {
    grid: color("--mapa-grade"),
    axis: color("--mapa-eixo"),
    body: color("--mapa-corpo"),
    halo: color("--mapa-halo"),
    cameras: MAP_CAMERA_TOKENS.map(color),
  };
}

if (typeof module !== "undefined") module.exports = {mapPosition, mapTheme};
