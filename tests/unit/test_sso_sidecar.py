"""Contract tests for the project-level didi_sso wiring.

The company package is available from the internal package index, not the
public PyPI index used by this test environment. A narrow fake checks our
integration contract without reimplementing or testing the SDK itself.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from fastapi import APIRouter


class _FakeSsoService:
    init_kwargs: dict[str, object]

    def __init__(self, **kwargs: object) -> None:
        type(self).init_kwargs = kwargs


class _FakeMcpAuthService:
    init_kwargs: dict[str, object]

    def __init__(self, **kwargs: object) -> None:
        type(self).init_kwargs = kwargs


class _FakeUser:
    def __init__(self, ldap: str, user_name: str) -> None:
        self.ldap = ldap
        self.user_name = user_name


def _load_sidecar(monkeypatch):
    fake_sdk = types.ModuleType("didi_sso")
    fake_sdk.SsoService = _FakeSsoService
    fake_sdk.McpAuthService = _FakeMcpAuthService
    fake_sdk.SsoMiddleware = type("SsoMiddleware", (), {})
    fake_sdk.User = _FakeUser
    fake_sdk.get_user = lambda: None
    fake_sdk.create_sso_router = lambda service, prefix: APIRouter(prefix=prefix)
    monkeypatch.setitem(sys.modules, "didi_sso", fake_sdk)
    monkeypatch.setenv("SSO_APP_ID", "forecast-app")
    monkeypatch.setenv("SSO_APP_KEY", "secret")
    monkeypatch.setenv("SSO_HOST", "https://sso.example.test")
    monkeypatch.setenv("SSO_DOMAIN", ".example.test")
    monkeypatch.setenv("MCP_APP_ID", "mcp-app")
    monkeypatch.setenv("MCP_SECRET_KEY", "mcp-secret")
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("SSO_DEFAULT_LDAP", "zhangsan")
    monkeypatch.setenv("SSO_DEFAULT_USER_NAME", "Zhang San")

    module_path = Path(__file__).parents[2] / "deploy" / "sso_sidecar" / "main.py"
    spec = importlib.util.spec_from_file_location("test_sso_sidecar", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_didi_sso_service_receives_standard_settings(monkeypatch) -> None:
    sidecar = _load_sidecar(monkeypatch)

    assert _FakeSsoService.init_kwargs == {
        "app_id": "forecast-app",
        "app_key": "secret",
        "sso_host": "https://sso.example.test",
        "domain": ".example.test",
        "login_path": "/auth/sso/login",
        "logout_path": "/auth/ldap/logout",
        "check_ticket_path": "/auth/sso/api/check_ticket",
        "check_code_path": "/auth/sso/api/check_code",
        "mock": False,
        "user_index_path": "/auth/api/user/index",
        "upm_check_user_ticket_path": "/auth/sso/api/get_user_by_ticket",
    }
    assert _FakeMcpAuthService.init_kwargs == {
        "mcp_app_id": "mcp-app",
        "mcp_secret_key": "mcp-secret",
    }
    assert sidecar.default_user.ldap == "zhangsan"


def test_streamlit_websocket_url_preserves_base_path(monkeypatch) -> None:
    sidecar = _load_sidecar(monkeypatch)
    monkeypatch.setattr(sidecar, "STREAMLIT_URL", "https://streamlit.example.test/ui")

    assert sidecar._streamlit_websocket_url("session=1") == (
        "wss://streamlit.example.test/ui/_stcore/stream?session=1"
    )


def test_user_header_uses_didi_sso_context(monkeypatch) -> None:
    sidecar = _load_sidecar(monkeypatch)
    monkeypatch.setattr(sidecar, "get_user", lambda: _FakeUser("ZhangSan", "Zhang San"))

    assert sidecar._get_user_ldap() == "zhangsan"
