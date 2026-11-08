import pandas as pd
import pytest
from autogluon_agent.errors import AgaError
from autogluon_agent.profile import infer_task, profile
from autogluon_agent.io import read_table

def test_infer_task(make_binary, make_regression, make_multiclass):
    assert infer_task(make_binary()["target"]) == "binary"
    assert infer_task(make_regression()["target"]) == "regression"
    assert infer_task(make_multiclass()["target"]) == "multiclass"

def test_profile_kinds_and_suggestions(make_binary):
    df = make_binary()
    df["row_id"] = range(len(df))
    p = profile(df, "target")
    kinds = {c["name"]: c["kind"] for c in p["columns"]}
    assert kinds["ts"] == "datetime"
    assert kinds["row_id"] == "id_like"
    assert kinds["cat"] == "categorical"
    assert "group_id" in p["suggestions"]["group_candidates"]
    assert "ts" in p["suggestions"]["time_candidates"]

def test_read_semicolon_comma_decimal(tmp_path):
    f = tmp_path / "eu.csv"
    f.write_text("a;b\n1,5;x\n2,25;y\n", encoding="utf-8")
    df = read_table(f)
    assert list(df.columns) == ["a", "b"] and df["a"].tolist() == [1.5, 2.25]

def test_read_parquet(tmp_path, make_binary):
    f = tmp_path / "d.parquet"; make_binary().to_parquet(f)
    assert read_table(f).shape == make_binary().shape

def test_read_mixed_comma_decimal_column(tmp_path):
    f = tmp_path / "mixed.csv"
    f.write_text("a;b\n1,5;x\n2;y\n3,25;z\n", encoding="utf-8")
    assert read_table(f)["a"].tolist() == [1.5, 2.0, 3.25]

def test_read_tab_delimited_comma_decimal(tmp_path):
    f = tmp_path / "eu.txt"
    f.write_text("a\tb\n1,5\tx\n2,25\ty\n", encoding="utf-8")
    df = read_table(f)
    assert list(df.columns) == ["a", "b"] and df["a"].tolist() == [1.5, 2.25]

def test_read_table_errors(tmp_path):
    bad = tmp_path / "d.xlsx"; bad.write_text("x")
    with pytest.raises(AgaError):
        read_table(bad)
    with pytest.raises(AgaError):
        read_table(tmp_path / "missing.csv")

def test_long_unique_strings_are_text():
    n = 300
    df = pd.DataFrame({
        "doc": [f"free text document number {i:04d} with some extra words" for i in range(n)],
        "target": [i % 2 for i in range(n)],
    })
    kinds = {c["name"]: c["kind"] for c in profile(df, "target")["columns"]}
    assert kinds["doc"] == "text"

def test_read_comma_delimited_quoted_comma_decimal(tmp_path):
    f = tmp_path / "q.csv"
    f.write_text('id,code,target\n1,"1,5",0\n2,3,1\n3,"2,25",0\n', encoding="utf-8")
    df = read_table(f)
    assert list(df.columns) == ["id", "code", "target"] and len(df) == 3

def test_uuid_strings_are_id_like():
    import uuid
    n = 300
    df = pd.DataFrame({
        "key": [str(uuid.UUID(int=i * 7919 + 12345)) for i in range(n)],
        "target": [i % 2 for i in range(n)],
    })
    kinds = {c["name"]: c["kind"] for c in profile(df, "target")["columns"]}
    assert kinds["key"] == "id_like"
