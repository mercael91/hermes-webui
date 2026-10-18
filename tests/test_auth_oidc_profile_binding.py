import pytest

from api import auth_oidc


def _claims(groups):
    return {"sub": "user-1", "email": "user@example.com", "groups": groups}


def test_no_profile_binding_when_unset():
    assert (
        auth_oidc._resolve_bound_profile(
            _claims(["hermes-alice"]), profile_claim="", profile_map={}
        )
        is None
    )


def test_single_mapped_group_binds_profile():
    assert (
        auth_oidc._resolve_bound_profile(
            _claims(["hermes-alice", "other"]),
            profile_claim="groups",
            profile_map={"hermes-alice": "alice", "hermes-bob": "bob"},
        )
        == "alice"
    )


def test_no_match_is_rejected():
    with pytest.raises(auth_oidc.OIDCAuthError) as exc:
        auth_oidc._resolve_bound_profile(
            _claims(["unmapped"]),
            profile_claim="groups",
            profile_map={"hermes-alice": "alice"},
        )
    assert exc.value.status_code == 403


def test_ambiguous_multi_profile_match_is_rejected():
    with pytest.raises(auth_oidc.OIDCAuthError) as exc:
        auth_oidc._resolve_bound_profile(
            _claims(["hermes-alice", "hermes-bob"]),
            profile_claim="groups",
            profile_map={"hermes-alice": "alice", "hermes-bob": "bob"},
        )
    assert exc.value.status_code == 403


def test_missing_claim_is_rejected():
    with pytest.raises(auth_oidc.OIDCAuthError) as exc:
        auth_oidc._resolve_bound_profile(
            {"sub": "user-1"},
            profile_claim="groups",
            profile_map={"hermes-alice": "alice"},
        )
    assert exc.value.status_code == 403


def test_profile_map_accepts_json_string():
    assert auth_oidc._normalize_profile_map('{"hermes-alice": "alice"}') == {
        "hermes-alice": "alice"
    }


def test_profile_map_rejects_invalid_json():
    with pytest.raises(auth_oidc.OIDCConfigError):
        auth_oidc._normalize_profile_map("not-json")


def test_require_profile_binding_requires_claim_and_map(monkeypatch):
    monkeypatch.setenv("HERMES_WEBUI_OIDC_ISSUER", "https://issuer.example.com")
    monkeypatch.setenv("HERMES_WEBUI_OIDC_CLIENT_ID", "client")
    monkeypatch.setenv("HERMES_WEBUI_OIDC_ALLOW_CLAIM", "groups")
    monkeypatch.setenv("HERMES_WEBUI_OIDC_ALLOW_VALUES", "hermes-users")
    monkeypatch.setenv("HERMES_WEBUI_OIDC_REQUIRE_PROFILE_BINDING", "true")
    monkeypatch.delenv("HERMES_WEBUI_OIDC_PROFILE_CLAIM", raising=False)
    monkeypatch.delenv("HERMES_WEBUI_OIDC_PROFILE_MAP", raising=False)
    with pytest.raises(auth_oidc.OIDCConfigError):
        auth_oidc._require_oidc_config()


def test_require_profile_binding_accepts_configured_binding(monkeypatch):
    monkeypatch.setenv("HERMES_WEBUI_OIDC_ISSUER", "https://issuer.example.com")
    monkeypatch.setenv("HERMES_WEBUI_OIDC_CLIENT_ID", "client")
    monkeypatch.setenv("HERMES_WEBUI_OIDC_ALLOW_CLAIM", "groups")
    monkeypatch.setenv("HERMES_WEBUI_OIDC_ALLOW_VALUES", "hermes-users")
    monkeypatch.setenv("HERMES_WEBUI_OIDC_REQUIRE_PROFILE_BINDING", "true")
    monkeypatch.setenv("HERMES_WEBUI_OIDC_PROFILE_CLAIM", "groups")
    monkeypatch.setenv("HERMES_WEBUI_OIDC_PROFILE_MAP", '{"hermes-alice": "alice"}')
    cfg = auth_oidc._require_oidc_config()
    assert cfg["require_profile_binding"] is True
    assert cfg["profile_claim"] == "groups"
    assert cfg["profile_map"] == {"hermes-alice": "alice"}


def test_normalize_bool_variants():
    assert auth_oidc._normalize_bool(None) is False
    assert auth_oidc._normalize_bool(True) is True
    assert auth_oidc._normalize_bool("true") is True
    assert auth_oidc._normalize_bool("1") is True
    assert auth_oidc._normalize_bool("no") is False
