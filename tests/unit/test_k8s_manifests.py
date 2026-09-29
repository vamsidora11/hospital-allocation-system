"""Not a substitute for `kubectl apply --dry-run` against a real cluster (no cluster
available in this sandbox — see CLAUDE.md / PROGRESS.md Phase 11), but catches YAML syntax
errors and missing required fields before they'd fail in one."""

from pathlib import Path

import pytest
import yaml

MANIFESTS_DIR = Path(__file__).resolve().parents[2] / "deploy" / "k8s"
EXPECTED_KINDS = {
    "deployment.yaml": "Deployment",
    "service.yaml": "Service",
    "configmap.yaml": "ConfigMap",
    "secret.yaml": "Secret",
    "hpa.yaml": "HorizontalPodAutoscaler",
}


@pytest.mark.parametrize("filename,expected_kind", EXPECTED_KINDS.items())
def test_manifest_is_valid_yaml_with_expected_kind(filename, expected_kind):
    path = MANIFESTS_DIR / filename
    assert path.exists(), f"missing {filename}"

    docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    assert len(docs) == 1
    doc = docs[0]

    assert doc["apiVersion"]
    assert doc["kind"] == expected_kind
    assert doc["metadata"]["name"]


def test_secret_template_has_no_obviously_real_credentials():
    text = (MANIFESTS_DIR / "secret.yaml").read_text(encoding="utf-8")
    # cheap guard against accidentally committing a real key shaped like the ones we've
    # actually seen this session (long base64-ish strings) instead of the REPLACE_ME markers
    for line in text.splitlines():
        if ":" in line and "REPLACE_ME" not in line and not line.strip().startswith("#"):
            value = line.split(":", 1)[1].strip().strip('"')
            assert len(value) < 40, f"suspiciously long non-placeholder value: {line}"
