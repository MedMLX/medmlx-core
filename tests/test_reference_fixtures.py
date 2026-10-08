"""Versioned fixture provenance, complete coverage, and parser properties."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from fixture_cases import (
    FIXTURES,
    REFERENCE,
    FixtureMetadata,
    ReferenceSpec,
    array_cases,
    load_fixture,
)
from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.mark.parametrize("name", ["layout", "ops", "upsample", "precision"])
def test_recorded_helper_coverage_and_seeded_inputs(name):
    metadata, arrays = load_fixture(name)
    seeded, cases = array_cases()
    assert metadata.seed == 7081
    assert {record.id for record in metadata.cases} == {
        case.id for case in cases[name] if case.error is None
    }
    for key, value in seeded.items():
        np.testing.assert_array_equal(arrays[key], value)
    for record in metadata.cases:
        for key in record.outputs:
            assert key in arrays
            assert np.isfinite(arrays[key]).all()


def test_every_archive_has_current_provenance_and_is_small():
    paths = sorted(FIXTURES.parent.rglob("*.npz"))
    assert len(paths) >= 7
    for path in paths:
        assert path.stat().st_size < 80_000
        with np.load(path, allow_pickle=False) as archive:
            metadata = FixtureMetadata.from_json(str(archive["metadata"]))
        assert metadata.upstream == REFERENCE
        assert path.parent == FIXTURES


def test_missing_reference_fails_with_regeneration_hint():
    with pytest.raises(FileNotFoundError, match="run the README fixture commands"):
        load_fixture("unrecorded-helper")


def test_fixture_metadata_rejects_undeclared_fields():
    metadata, _ = load_fixture("layout")
    payload = json.loads(metadata.to_json())
    payload["unexpected"] = "not part of the schema"
    with pytest.raises(ValueError, match="undeclared fields"):
        FixtureMetadata.from_json(json.dumps(payload))


@given(st.text(), st.text())
@settings(database=None)
def test_reference_spec_round_trip(monai_version, torch_version):
    value = ReferenceSpec(
        1,
        monai_version or "1",
        REFERENCE.monai_revision,
        torch_version or "1",
        REFERENCE.torch_revision,
    )
    assert ReferenceSpec.from_json(value.to_json()) == value


JSON_VALUES = st.recursive(
    st.none() | st.booleans() | st.integers() | st.text(),
    lambda children: (
        st.lists(children, max_size=5) | st.dictionaries(st.text(), children, max_size=5)
    ),
    max_leaves=20,
)


@given(JSON_VALUES)
@settings(database=None)
def test_reference_parser_accepts_typed_objects_or_raises_value_error(payload):
    try:
        value = ReferenceSpec.from_json(json.dumps(payload))
    except ValueError:
        return
    assert ReferenceSpec.from_json(value.to_json()) == value


@given(JSON_VALUES)
@settings(database=None)
def test_fixture_parser_accepts_typed_objects_or_raises_value_error(payload):
    try:
        value = FixtureMetadata.from_json(json.dumps(payload))
    except ValueError:
        return
    assert isinstance(value.upstream, ReferenceSpec)
    assert value.platform == "darwin-arm64"
    assert type(value.seed) is int


@pytest.mark.parametrize(
    "field",
    ["schema_version", "monai_version", "monai_revision", "torch_version", "torch_revision"],
)
def test_reference_parser_rejects_invalid_field_types(field):
    payload = asdict(REFERENCE)
    payload[field] = None
    with pytest.raises(ValueError):
        ReferenceSpec.from_json(json.dumps(payload))


def test_reference_pin_matches_development_dependencies():
    import tomllib
    from pathlib import Path

    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    dependencies = project["project"]["optional-dependencies"]["dev"]
    assert f"monai=={REFERENCE.monai_version}" in dependencies
    assert f"torch=={REFERENCE.torch_version}" in dependencies


@given(st.integers(min_value=0), st.sampled_from([None, "constant", "gaussian"]))
@settings(database=None)
def test_fixture_metadata_round_trip(seed, mode):
    value = FixtureMetadata(
        REFERENCE,
        "darwin-arm64",
        seed,
        (),
        "torch CPU / NumPy host",
        np.__version__,
        REFERENCE.torch_version,
        mode,
    )
    assert FixtureMetadata.from_json(value.to_json()) == value


@pytest.mark.parametrize("schema", [True, False, 0, 1, "2", None])
def test_fixture_parser_rejects_unsupported_schema(schema):
    value = FixtureMetadata(
        REFERENCE,
        "darwin-arm64",
        7081,
        (),
        "torch CPU / NumPy host",
        np.__version__,
        REFERENCE.torch_version,
        None,
    )
    payload = json.loads(value.to_json())
    payload["schema_version"] = schema
    with pytest.raises(ValueError, match="schema"):
        FixtureMetadata.from_json(json.dumps(payload))
