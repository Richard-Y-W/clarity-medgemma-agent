from clarity.eval.soap_metrics import parse_headers, plan_bullets_ok

def test_parse_headers_ok():
    s = """SUBJECTIVE: chest pain
OBJECTIVE: BP 120/80
ASSESSMENT: possible angina
PLAN:
- ECG
- Troponin
"""
    sections, ok = parse_headers(s)
    assert ok is True
    assert "PLAN:" in sections
    assert plan_bullets_ok(sections["PLAN:"]) is True

def test_parse_headers_fail_missing_plan():
    s = """SUBJECTIVE: x
OBJECTIVE: y
ASSESSMENT: z
"""
    sections, ok = parse_headers(s)
    assert ok is False
