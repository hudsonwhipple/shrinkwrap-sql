import pytest

from shrinkwrap.prompts import build_messages, extract_sql, render_prompt


@pytest.mark.parametrize(
    "text, expected",
    [
        ("```sql\nSELECT 1;\n```", "SELECT 1"),
        ("```SQL\nSELECT 1\n```\nThis selects one.", "SELECT 1"),
        ("Here you go:\n```\nSELECT a\nFROM t\n```", "SELECT a\nFROM t"),
        ("<think>\nmaybe SELECT 2\n</think>\n\n```sql\nSELECT 1\n```", "SELECT 1"),
        ("SELECT 1;", "SELECT 1"),
        ("```sql\nSELECT 1", "SELECT 1"),  # cut off at max_tokens
        ("<think>never finished", ""),
    ],
)
def test_extract_sql(text, expected):
    assert extract_sql(text) == expected


def test_build_messages():
    msgs = build_messages("CREATE TABLE t (a);", "How many?", "SELECT count(*) FROM t")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "CREATE TABLE t (a);" in msgs[1]["content"] and "How many?" in msgs[1]["content"]
    assert extract_sql(msgs[2]["content"]) == "SELECT count(*) FROM t"
    assert len(build_messages("s", "q")) == 2


def test_inference_prompt_is_prefix_of_training_example():
    """Catches the classic chat-template bug: training and inference formats drifting apart."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")
    # mlx_lm's ChatDataset renders training examples exactly like this
    train = tok.apply_chat_template(build_messages("CREATE TABLE t (a);", "q", "SELECT a FROM t"), return_dict=False)
    prompt = render_prompt(tok, build_messages("CREATE TABLE t (a);", "q"))
    assert train[: len(prompt)] == prompt
