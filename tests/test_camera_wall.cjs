"use strict";
const test = require("node:test"), assert = require("node:assert/strict");
const {cameraWallLayout} = require("../src/custody_watch/assets/camera_wall.js");

const cameras = Object.freeze([
  Object.freeze({camera_id:"C1",floor_id:"térreo",segment_id:"s1"}),
  Object.freeze({camera_id:"C2",floor_id:"térreo",segment_id:"s2"}),
  Object.freeze({camera_id:"C3",floor_id:"superior",segment_id:"s3"}),
  Object.freeze({camera_id:"C4",floor_id:"térreo",segment_id:"s4"}),
]);

test("mosaico conserva ordem do cadastro e seleção sem mutar câmeras", () => {
  const result = cameraWallLayout(cameras,"térreo","C2","mosaic");
  assert.equal(result.selected,"C2");
  assert.deepEqual(result.cameras,[cameras[0],cameras[1],cameras[3]]);
  assert.equal(cameras[0].camera_id,"C1");
});
test("foco traz seleção para a primeira posição mantendo namespace e demais câmeras", () => {
  const result = cameraWallLayout(cameras,"térreo","C4","focus");
  assert.equal(result.selected,"C4");
  assert.deepEqual(result.cameras,[cameras[3],cameras[0],cameras[1]]);
  assert.equal(result.cameras[0],cameras[3]);
  assert.equal(result.cameras[0].segment_id,"s4");
});
test("mudar andar no foco recusa seleção fora do andar", () => {
  const result = cameraWallLayout(cameras,"superior","C2","focus");
  assert.equal(result.selected,"C3");
  assert.deepEqual(result.cameras,[cameras[2]]);
});
test("seleção removida ou inexistente volta à primeira câmera visível", () => {
  for (const selection of [null,"inexistente","C3"]) {
    const result = cameraWallLayout(cameras,"térreo",selection,"focus");
    assert.equal(result.selected,"C1");
    assert.deepEqual(result.cameras,[cameras[0],cameras[1],cameras[3]]);
  }
});
test("andar vazio não reutiliza câmera de outro andar", () => {
  for (const mode of ["mosaic","focus"]) {
    assert.deepEqual(cameraWallLayout(cameras,"vazio","C1",mode),{selected:null,cameras:[]});
  }
});
test("voltar ao mosaico restaura a ordem sem perder a câmera selecionada", () => {
  const focused = cameraWallLayout(cameras,"térreo","C2","focus");
  const mosaic = cameraWallLayout(cameras,"térreo",focused.selected,"mosaic");
  assert.equal(mosaic.selected,"C2");
  assert.deepEqual(mosaic.cameras,[cameras[0],cameras[1],cameras[3]]);
  assert.throws(() => cameraWallLayout(cameras,"térreo","C2","live"),RangeError);
});
