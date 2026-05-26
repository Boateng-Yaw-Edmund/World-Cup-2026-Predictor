from __future__ import annotations

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.pipeline import SetupError, WorldCupModelPipeline
from backend.simulator import GROUPS_2026, compare_models, simulate_tournament


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = WorldCupModelPipeline(PROJECT_ROOT)


class ApiHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/status":
            self.write_json({"ok": True, **PIPELINE.status()})
            return
        if path == "/api/groups":
            self.write_json({"ok": True, "groups": GROUPS_2026})
            return
        if path == "/":
            self.write_static(PROJECT_ROOT / "index.html")
            return
        static_path = self.resolve_static_path(path)
        if static_path:
            self.write_static(static_path)
            return
        self.write_json({"ok": False, "error": "Not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = self.read_body()
        model = body.get("model", "rf")

        try:
            if path == "/api/train":
                PIPELINE.train()
                self.write_json({"ok": True, "status": PIPELINE.status()})
                return
            if path == "/api/predict":
                home = body["home"]
                away = body["away"]
                prediction = PIPELINE.predict_match(home, away, model)
                self.write_json({"ok": True, "prediction": prediction.__dict__, "status": PIPELINE.status()})
                return
            if path == "/api/simulate":
                result = simulate_tournament(
                    PIPELINE,
                    model,
                    include_group_matches=bool(body.get("include_group_matches", False)),
                )
                self.write_json({"ok": True, "simulation": result})
                return
            if path == "/api/compare":
                comparison = compare_models(PIPELINE)
                self.write_json({"ok": True, "comparison": comparison})
                return
        except SetupError as exc:
            self.write_json({"ok": False, "error": str(exc), "status": PIPELINE.status()}, status=424)
            return
        except KeyError as exc:
            self.write_json({"ok": False, "error": f"Missing field: {exc}"}, status=400)
            return
        except Exception as exc:
            self.write_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)
            return

        self.write_json({"ok": False, "error": "Not found"}, status=404)

    def read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def write_json(self, payload: dict, status: int = 200) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def write_static(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.write_json({"ok": False, "error": "Not found"}, status=404)
            return

        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "text/javascript"
        self.send_response(200)
        self.send_cors_headers()
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def resolve_static_path(self, request_path: str) -> Path | None:
        allowed = {
            "/index.html": PROJECT_ROOT / "index.html",
            "/app.js": PROJECT_ROOT / "app.js",
            "/styles.css": PROJECT_ROOT / "styles.css",
            "/favicon.ico": PROJECT_ROOT / "favicon.ico",
        }
        return allowed.get(request_path)

    def send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    host = "127.0.0.1"
    port = 8768
    server = ThreadingHTTPServer((host, port), ApiHandler)
    print(f"World Cup predictor API running on http://{host}:{port}")
    print(f"UI available at http://{host}:{port}/")
    print("Endpoints: GET /api/status, POST /api/train, POST /api/predict, POST /api/simulate, POST /api/compare")
    server.serve_forever()


if __name__ == "__main__":
    main()
