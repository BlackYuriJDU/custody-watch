import json
import shutil
import subprocess
from pathlib import Path

from custody_watch.camera_registry import load_camera_registry
from custody_watch.contact_review_queue import prepare_contact_review_queue
from custody_watch.multicamera_view import ReplayView
from tests.test_multicamera_media import make_video_registry


def test_fila_do_gerador_python_e_vinculada_pelo_painel_ao_snapshot_real(tmp_path):
    path, _ = make_video_registry(tmp_path)
    registry = load_camera_registry(path)
    queue = prepare_contact_review_queue(registry, seed=7, frames_per_camera=2, min_gap_s=0)
    first = queue["cameras"][0]["frames"][0]
    view = ReplayView(registry)
    payload = {
        "text": json.dumps(queue),
        "config": view.config(),
        "snapshot": view.snapshot(first["global_t"])["cameras"][0],
    }
    node = shutil.which("node")
    assert node, "Interoperabilidade exige Node.js, também presente na CI."
    module = Path(__file__).parents[1] / "src/custody_watch/assets/review_queue.js"
    script = """
const r = require(process.argv[1]);
const data = JSON.parse(require('node:fs').readFileSync(0,'utf8'));
const queue = r.parseQueue(data.text,data.config), cursor = new r.QueueCursor(queue);
const context = r.matchFrame(cursor.current,data.config,data.config.cameras[0],data.snapshot);
process.stdout.write(JSON.stringify({item:cursor.current,ids:context.tracks.map(t=>t.local_track_id),progress:cursor.progress(null)}));
"""
    result = subprocess.run(
        [node, "-e", script, str(module)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=15,
    )
    value = json.loads(result.stdout)
    assert value["item"]["frame_index"] == first["frame_index"]
    assert value["ids"] == first["local_track_ids"]
    assert value["item"]["registry_sha256"] == registry.manifest_sha256
    assert value["progress"]["observations_done"] == 0
    assert value["progress"]["frames_total"] == 2


def test_json_malformado_com_muitas_aspas_nao_bloqueia_parser():
    node = shutil.which("node")
    assert node, "Interoperabilidade exige Node.js, também presente na CI."
    module = Path(__file__).parents[1] / "src/custody_watch/assets/review_queue.js"
    script = """
const {parseQueue} = require(process.argv[1]);
const input = String.fromCharCode(34) + String.fromCharCode(92,34).repeat(300000);
try { parseQueue(input, {}); process.exitCode=1; }
catch(error) {
  if(error.message !== 'Arquivo JSON inválido.') throw error;
  process.stdout.write('rejected');
}
"""
    result = subprocess.run(
        [node, "-e", script, str(module)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=15,
    )
    assert result.stdout == "rejected"
