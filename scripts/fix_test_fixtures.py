from pathlib import Path
p=Path('backend/app/reasoning/ocr.py');s=p.read_text().replace('OCR is unavailable. Install and configure the backend PaddleOCR engine.',"I couldn't read the text because OCR is unavailable.");p.write_text(s)
p=Path('backend/tests/test_behavior_and_voice_control.py');s=p.read_text().replace('def test_13_vlm_comprehensive_scene_description():','''def test_13_vlm_comprehensive_scene_description(monkeypatch):
    # Test successful provider response independently of a running Ollama server.
    class Response:
        status_code = 200
        def json(self):
            return {"response": "The scene shows a plain gray surface filling the view."}
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: Response())''');p.write_text(s)
