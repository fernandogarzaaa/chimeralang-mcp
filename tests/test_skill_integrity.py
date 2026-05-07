"""test_skill_integrity.py — structural integrity checks for the chimera skill files."""
from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).parent.parent
SKILL_MD = REPO / ".claude" / "skills" / "chimera" / "SKILL.md"
AGENTS_MD = REPO / "AGENTS.md"
PYPROJECT = REPO / "pyproject.toml"
SERVER_PY = REPO / "chimeralang_mcp" / "server.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract YAML-ish key: value pairs from a --- fenced frontmatter block."""
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}
    result: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip().strip('"')
    return result


def _extract_chimera_tool_names(text: str) -> set[str]:
    """Return every chimera_* identifier mentioned in backticks."""
    return set(re.findall(r"`(chimera_[a-z_]+)`", text))


def _extract_routing_table_rows(text: str) -> list[str]:
    """Return the raw content of every routing-matrix table row (pipes stripped)."""
    rows = []
    for line in text.splitlines():
        # Table rows contain at least two | separators and a chimera_ reference
        if line.count("|") >= 2 and "chimera_" in line:
            rows.append(line.strip())
    return rows


class TestFrontmatter(unittest.TestCase):
    def setUp(self):
        self.fm = _parse_frontmatter(_read(SKILL_MD))

    def test_name_is_chimera(self):
        self.assertEqual(self.fm.get("name"), "chimera")

    def test_chimera_version_present(self):
        self.assertIn("chimera_version", self.fm, "frontmatter must declare chimera_version")

    def test_chimera_version_matches_pyproject(self):
        with open(PYPROJECT, "rb") as f:
            data = tomllib.load(f)
        pkg_version = data["project"]["version"]
        self.assertEqual(
            self.fm["chimera_version"],
            pkg_version,
            f"SKILL.md chimera_version ({self.fm['chimera_version']!r}) != "
            f"pyproject.toml version ({pkg_version!r})",
        )

    def test_description_present_and_nonempty(self):
        desc = self.fm.get("description", "")
        self.assertTrue(len(desc) > 20, "frontmatter description should be non-trivial")


class TestToolNamesExistInServer(unittest.TestCase):
    def test_all_skill_tools_referenced_in_server(self):
        skill_tools = _extract_chimera_tool_names(_read(SKILL_MD))
        server_text = _read(SERVER_PY)
        missing = [t for t in sorted(skill_tools) if t not in server_text]
        self.assertEqual(
            missing, [],
            f"Tools mentioned in SKILL.md but absent from server.py: {missing}",
        )

    def test_all_agents_tools_referenced_in_server(self):
        agents_tools = _extract_chimera_tool_names(_read(AGENTS_MD))
        server_text = _read(SERVER_PY)
        missing = [t for t in sorted(agents_tools) if t not in server_text]
        self.assertEqual(
            missing, [],
            f"Tools mentioned in AGENTS.md but absent from server.py: {missing}",
        )


class TestRoutingMatrixDrift(unittest.TestCase):
    """Ensure the primary routing matrices in SKILL.md and AGENTS.md stay in sync."""

    def _primary_tools(self, text: str) -> set[str]:
        """Tools appearing in the first routing table (before the reasoning lane)."""
        tools: set[str] = set()
        in_primary = False
        for line in text.splitlines():
            if "## Routing matrix" in line:
                in_primary = True
            elif in_primary and line.startswith("## "):
                break
            elif in_primary:
                tools.update(re.findall(r"`(chimera_[a-z_]+)`", line))
        return tools

    def test_primary_routing_tables_match(self):
        skill_tools = self._primary_tools(_read(SKILL_MD))
        agents_tools = self._primary_tools(_read(AGENTS_MD))
        in_skill_only = skill_tools - agents_tools
        in_agents_only = agents_tools - skill_tools
        self.assertEqual(
            in_skill_only, set(),
            f"Tools in SKILL.md primary matrix but not AGENTS.md: {in_skill_only}",
        )
        self.assertEqual(
            in_agents_only, set(),
            f"Tools in AGENTS.md primary matrix but not SKILL.md: {in_skill_only}",
        )

    def test_agents_version_comment_matches_pyproject(self):
        with open(PYPROJECT, "rb") as f:
            data = tomllib.load(f)
        pkg_version = data["project"]["version"]
        agents_text = _read(AGENTS_MD)
        self.assertIn(
            f"chimera_version: {pkg_version}",
            agents_text,
            f"AGENTS.md version comment must contain 'chimera_version: {pkg_version}'",
        )


class TestSkillStructure(unittest.TestCase):
    def setUp(self):
        self.skill = _read(SKILL_MD)
        self.agents = _read(AGENTS_MD)

    def test_skill_has_required_sections(self):
        required = [
            "## Decision tree",
            "## Routing matrix",
            "## Calling conventions",
            "## Reasoning lane",
            "## Glyph lane",
            "## Mode selection",
            "## Worked examples",
            "## Enforcement checklist",
            "## Telemetry reaction matrix",
            "## Skip conditions",
            "## Long tail",
        ]
        for section in required:
            self.assertIn(section, self.skill, f"SKILL.md missing section: {section!r}")

    def test_agents_has_required_sections(self):
        required = [
            "## Routing matrix",
            "## Calling conventions",
            "## Reasoning lane",
            "## Glyph lane",
            "## Mode selection",
            "## Worked examples",
            "## Enforcement checklist",
            "## Telemetry reaction matrix",
        ]
        for section in required:
            self.assertIn(section, self.agents, f"AGENTS.md missing section: {section!r}")

    def test_skill_mentions_codex_compatibility(self):
        self.assertIn("Codex", self.skill)

    def test_agents_points_back_to_skill(self):
        self.assertIn("SKILL.md", self.agents)


if __name__ == "__main__":
    unittest.main()
