from pathlib import Path
p=Path('backend/app/services/pipeline.py');s=p.read_text();s=s.replace('''        if frame is None:

            frame = (
                self.get_annotated_frame()
            )

''','');s=s.replace('"status": "ON_DEMAND",\n                "engine_loaded"','"status": "READY" if self.ocr.ocr_engine is not None else "UNAVAILABLE",\n                "engine_loaded"');p.write_text(s)
p=Path('backend/app/reasoning/vlm.py');s=p.read_text();a=s.index('        # Intelligent structured local fallback');s=s[:a]+'''        return "Visual AI is unavailable. Check the backend Ollama service and model."
''';s=s.replace('Using local structured fallback reasoner.','Visual AI is unavailable.');p.write_text(s)
p=Path('backend/app/modes/ask.py');s=p.read_text();s=s.replace('"success": True','"success": not answer.startswith(("Visual AI is unavailable", "Visual scene analysis took too long", "Unable to process camera image"))');p.write_text(s)
p=Path('backend/app/reasoning/ocr.py');s=p.read_text();a=s.index('            # Fallback test pattern');b=s.index('        t_inf_ms',a);s=s[:a]+s[b:];s=s.replace('Using CV fallback engine.','OCR is unavailable.');s=s.replace('Engine is None, using test fallback','Engine is unavailable');s=s.replace('msg = "I couldn\'t read the text clearly."','msg = "OCR is unavailable. Install and configure the backend PaddleOCR engine." if self.ocr_engine is None else "I couldn\'t read the text clearly."');p.write_text(s)
# The read-handler unit test must provide an OCR engine, rather than rely on fabricated runtime output.
p=Path('backend/tests/test_read_and_context.py');s=p.read_text();s=s.replace('def test_read_mode_ocr():','def test_read_mode_ocr(monkeypatch):');s=s.replace('    read_handler = ReadModeHandler(ocr_provider)','''    class TestOCREngine:
        def predict(self, image):
            return [{"rec_texts": ["VISIONMATE ASSISTANT", "AI Lab Room 302"], "rec_scores": [0.99, 0.98]}]
    monkeypatch.setattr(ocr_provider, "ocr_engine", TestOCREngine())
    read_handler = ReadModeHandler(ocr_provider)''');p.write_text(s)
