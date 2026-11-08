import pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from prepare import prepare

def test_prepare_hides_test_and_blanks_leaks(tmp_path, make_binary):
    df = make_binary()
    entry = dict(name="toy", target="target", task="binary", metric="roc_auc", goal="predict target",
                 test_split={"kind": "group", "column": "group_id"}, leak_columns=["leak"])
    prepare(entry, tmp_path, df=df)
    train = pd.read_csv(tmp_path / "workspace/toy/train.csv")
    hidden = tmp_path / ".hidden/toy"
    prod = pd.read_csv(hidden / "test_features_production.csv")
    naive = pd.read_csv(hidden / "test_features_naive.csv")
    assert not set(train.group_id) & set(pd.read_csv(hidden / "test.csv").group_id)
    assert prod["leak"].isna().all() and naive["leak"].notna().all()
    assert "target" not in prod.columns
    assert not any(p.name.startswith("test") for p in (tmp_path / "workspace/toy").iterdir())
    assert "roc_auc" in (tmp_path / "workspace/toy/task.md").read_text()
