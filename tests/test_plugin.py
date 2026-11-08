import json, re
from pathlib import Path
import yaml
from autogluon_agent.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills/autogluon-agent/SKILL.md"

def frontmatter(p):
    m = re.match(r"^---\n(.*?)\n---\n", p.read_text(encoding="utf-8"), re.S)
    return yaml.safe_load(m.group(1))

def test_manifests_parse():
    pj = json.loads((ROOT / ".claude-plugin/plugin.json").read_text())
    mj = json.loads((ROOT / ".claude-plugin/marketplace.json").read_text())
    assert pj["name"] == "autogluon-agent" and mj["plugins"][0]["name"] == "autogluon-agent"

def test_skill_frontmatter_and_size():
    fm = frontmatter(SKILL)
    assert fm["name"] == "autogluon-agent" and fm["description"].startswith("Use when")
    assert len(SKILL.read_text(encoding="utf-8").splitlines()) < 300

def test_every_aga_command_in_skill_exists():
    used = set(re.findall(r"`aga (\w+)", SKILL.read_text(encoding="utf-8")))
    known = set(build_parser()._subparsers._group_actions[0].choices)
    assert used and used <= known

def test_playbooks_do_not_mirror_benchmark_winning_features():
    banned = ["airport", "rotated", "hour_of_week", "hour of week", "opening_angle", "target opening"]
    for p in (SKILL, ROOT / "skills/autogluon-agent/references/features.md"):
        text = p.read_text(encoding="utf-8").lower()
        for term in banned:
            assert term not in text, f"{p.name} contains {term!r}"

def test_references_exist_and_do_not_name_benchmark_leaks():
    text = (ROOT / "skills/autogluon-agent/references/leakage.md").read_text(encoding="utf-8").lower()
    for banned in ["duration", "bank", "statsbomb", "adult", "casual", "registered"]:
        assert banned not in text
