from rag_mcp.scrub import scrub, scrub_text


def test_scrub_keys_and_tokens():
    text = (
        "aws AKIAIOSFODNN7EXAMPLE and sk-ant-api03-abc123def456ghi789 and "
        "ghp_abcdefghijklmnopqrstuv123456 and xoxb-1234567890-abcdefg"
    )
    out, counts = scrub(text)
    assert "AKIA" not in out
    assert "sk-ant" not in out
    assert "ghp_" not in out
    assert "xoxb" not in out
    assert counts["aws_key"] == 1


def test_scrub_pem_and_email():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----\nmail tyrelcb@gmail.com"
    out = scrub_text(text)
    assert "MIIabc" not in out
    assert "tyrelcb@gmail.com" not in out
    assert "[REDACTED:private_key]" in out
    assert "[REDACTED:email]" in out


def test_scrub_generic_assignment():
    out = scrub_text("export API_KEY=abcd1234efgh5678")
    assert "abcd1234efgh5678" not in out


def test_scrub_leaves_normal_text():
    text = "run pytest in the tests directory then check port 8004"
    assert scrub_text(text) == text
