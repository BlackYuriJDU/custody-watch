"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {parseQueue, matchFrame, QueueCursor, MAX_BYTES} = require("../src/custody_watch/assets/review_queue.js");
const {ReviewStore, createDraft} = require("../src/custody_watch/assets/contact_review.js");
const reg="a".repeat(64), seq="b".repeat(64);
function config() {
  return {registry_sha256:reg,timeline:{start_s:-10,end_s:10},cameras:[
    {camera_id:"a",segment_id:"seg",sequence_sha256:seq,provenance:"ground_truth",mode:"oracle",source_kind:"video",clock:{offset_s:-1}},
    {camera_id:"b",segment_id:"seg",sequence_sha256:seq,provenance:"ground_truth",mode:"oracle",source_kind:"video",clock:{offset_s:0}},
  ]};
}
function data() {
  return {schema_version:1,kind:"ground_contact_review_queue",registry_sha256:reg,
    selection:{algorithm:"sha256_rank_greedy_gap_v1",seed:7,frames_per_camera:6,min_gap_s:1,unit:"camera_frame"},
    cameras:[{camera_id:"a",segment_id:"seg",sequence_sha256:seq,prediction_provenance:"ground_truth",candidate_frames:4,selected_frames:2,frames:[
      {frame_index:0,media_t:1,global_t:0,local_track_ids:[1,2]},
      {frame_index:2,media_t:2,global_t:1,local_track_ids:[1]},
    ]},{camera_id:"b",segment_id:"seg",sequence_sha256:seq,prediction_provenance:"ground_truth",candidate_frames:1,selected_frames:1,frames:[
      {frame_index:0,media_t:0,global_t:0,local_track_ids:[3]},
    ]}],excluded_cameras:[]};
}
const parse = value => parseQueue(JSON.stringify(value),config());
function snapshot(item) {
  return {camera_id:item.camera_id,segment_id:item.segment_id,status:"observed",image:{
    registry_sha256:reg,sequence_sha256:seq,frame_index:item.frame_index,t:item.global_t,
    media_t:item.media_t,age_s:0,historical:false,kind:"video",tracks:item.local_track_ids.map(id=>({
      track_key:{camera_id:item.camera_id,segment_id:item.segment_id,local_track_id:id},
      cls:"person",bbox:[0,0,20,30],position_m:null,projected_position_m:null,
      position_status:"ground_contact_unknown",association_eligible:false,
    })),
  }};
}
test("fila é congelada e cursor troca de câmera e volta sem alterar origem",()=>{
  const input=data(), queue=parse(input), cursor=new QueueCursor(queue);
  input.cameras[0].frames[0].local_track_ids.push(9);
  assert.deepEqual(cursor.current.local_track_ids,[1,2]);
  assert.ok(Object.isFrozen(queue) && Object.isFrozen(cursor.current.local_track_ids));
  assert.equal(cursor.move(-1),false);
  assert.equal(cursor.index,0);
  assert.equal(cursor.move(1),true);
  assert.equal(cursor.current.frame_index,2);
  cursor.move(1); assert.equal(cursor.current.camera_id,"b");
  assert.equal(cursor.move(1),false);
  cursor.move(-1); assert.equal(cursor.current.camera_id,"a");
  assert.equal(cursor.current.prediction_provenance,"ground_truth");
});
test("progresso só conta registros guardados e preserva estimativa",()=>{
  const cursor=new QueueCursor(parse(data())), store=new ReviewStore(reg);
  assert.deepEqual(cursor.progress(store),{frames_total:3,frames_done:0,observations_total:4,observations_done:0});
  const ctx=matchFrame(cursor.current,config(),config().cameras[0],snapshot(cursor.current));
  const draft=createDraft(ctx,[100,100],1,{status:"unknown",point_px:null,provenance:"model_prediction",method:"Estimativa de teste."});
  store.add(draft);
  assert.equal(cursor.progress(store).observations_done,1);
  assert.equal(cursor.progress(store).frames_done,0);
  cursor.move(1);
  assert.equal(cursor.progress(store).observations_done,1);
  cursor.move(-1);
  assert.equal(store.get(cursor.current,1).provenance,"model_prediction");
  assert.equal(store.get(cursor.current,1).human_confirmed,false);
  store.remove(cursor.current,1);
  assert.equal(cursor.progress(store).observations_done,0);
});
for (const [name,modify] of [
  ["outro cadastro",d=>d.registry_sha256="c".repeat(64)],
  ["outra sequência",d=>d.cameras[0].sequence_sha256="c".repeat(64)],
  ["outro segmento",d=>d.cameras[0].segment_id="old"],
  ["origem divergente",d=>d.cameras[0].prediction_provenance="human_review"],
  ["câmera duplicada",d=>d.cameras.push(d.cameras[0])],
  ["quadro duplicado",d=>d.cameras[0].frames[1]=d.cameras[0].frames[0]],
  ["ID duplicado",d=>d.cameras[0].frames[0].local_track_ids=[1,1]],
  ["ID inseguro",d=>d.cameras[0].frames[0].local_track_ids=[2**53]],
  ["ID booleano",d=>d.cameras[0].frames[0].local_track_ids=[true]],
  ["seed inválida",d=>d.selection.seed=-1],
  ["versão desconhecida",d=>d.schema_version=2],
  ["contagem divergente",d=>d.cameras[0].selected_frames=1],
  ["tempo fora do relógio",d=>d.cameras[0].frames[0].global_t=8],
  ["campo extra",d=>d.path="C:/private/file"],
  ["gabarito como contato",d=>d.cameras[0].frames[0].point_px=[1,2]],
  ["câmera omitida",d=>d.cameras.pop()],
  ["gap inválido",d=>d.selection.min_gap_s=3],
  ["fila vazia",d=>d.cameras.forEach(c=>{c.frames=[];c.selected_frames=0;})],
]) test(`recusa ${name}`,()=>{const d=data();modify(d);assert.throws(()=>parse(d));});
test("JSON estrito recusa chave repetida, UTF-8 excessivo, overflow e profundidade",()=>{
  const raw=JSON.stringify(data());
  assert.throws(()=>parseQueue(raw.replace('"seed":7','"seed":7,"seed":8'),config()),/duplicada/);
  assert.throws(()=>parseQueue(raw.replace('"seed":7','"seed":7,"s\\u0065ed":8'),config()),/duplicada/);
  assert.throws(()=>parseQueue(" ".repeat(MAX_BYTES+1),config()),/limite/);
  assert.throws(()=>parseQueue("é".repeat(MAX_BYTES/2+1),config()),/limite/);
  assert.throws(()=>parseQueue('['.repeat(65)+'0'+']'.repeat(65),config()),/profundidade/);
  assert.throws(()=>parseQueue(raw.replace('"min_gap_s":1','"min_gap_s":1e999'),config()));
});
for(const [name,change] of [
  ["outro quadro",s=>s.image.frame_index++], ["imagem histórica errada",s=>s.image.t=1],
  ["ID ausente",s=>s.image.tracks.pop()], ["fonte offline",s=>s.image=null],
  ["quadro missing",s=>s.status="missing"], ["namespace divergente",s=>s.segment_id="old"],
  ["ID não é pessoa",s=>s.image.tracks[0].cls="suitcase"],
]) test(`revisão bloqueada: ${name}`,()=>{
  const item=new QueueCursor(parse(data())).current,s=snapshot(item);change(s);
  assert.throws(()=>matchFrame(item,config(),config().cameras[0],s));
});
test("limite total de IDs não é truncado e exclusões são conferidas",()=>{
  const d=data();d.cameras[0].frames[0].local_track_ids=Array.from({length:1001},(_,i)=>i);
  assert.throws(()=>parse(d),/limite/);
  const valid=data();valid.cameras.pop();valid.excluded_cameras=[{camera_id:"b",reason:"unavailable"}];
  assert.equal(parse(valid).items.length,2);
  valid.excluded_cameras[0].reason="guess";
  assert.throws(()=>parse(valid));
});
test("conclusão de quadros depende de todos os IDs e remoção desfaz progresso",()=>{
  const cursor=new QueueCursor(parse(data())), store=new ReviewStore(reg);
  for (const item of cursor.queue.items) {
    const camera=config().cameras.find(c=>c.camera_id===item.camera_id);
    const context=matchFrame(item,config(),camera,snapshot(item));
    for (const id of item.local_track_ids) store.add(createDraft(context,[100,100],id,{
      status:"unknown",point_px:null,provenance:"model_prediction",method:"Estimativa simulada.",
    }));
  }
  assert.deepEqual(cursor.progress(store),{frames_total:3,frames_done:3,observations_total:4,observations_done:4});
  store.remove(cursor.current,1);
  assert.deepEqual(cursor.progress(store),{frames_total:3,frames_done:2,observations_total:4,observations_done:3});
});
