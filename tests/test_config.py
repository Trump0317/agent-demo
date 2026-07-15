"""Tests for MVT 1.3 / Phase 4 — Config management (YAML support)."""
import os
import tempfile
from pathlib import Path

import pytest

from src.agent.config import Config
from src.agent.exceptions import ConfigError


def _write_temp_yaml(content: str) -> str:
    """Helper: write YAML content to a temp file, return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name


class TestConfigDefaults:
    """测试默认值"""

    def test_default_values(self):
        cfg = Config()
        assert cfg.llm_model == "deepseek-chat"
        assert cfg.llm_base_url == "https://api.deepseek.com/v1"
        assert cfg.llm_timeout == 60
        assert cfg.llm_max_retries == 2
        assert cfg.max_iterations == 10
        assert cfg.max_context_tokens == 64000
        assert cfg.tool_timeout == 30
        assert cfg.trace_enabled is True
        assert cfg.compression_threshold == 0.8


class TestFromEnv:
    """测试 from_env 加载"""

    def test_missing_api_key_raises(self, monkeypatch):
        """缺少 DEEPSEEK_API_KEY 应抛出 ConfigError"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="DEEPSEEK_API_KEY"):
            Config.from_env(env_file=None)

    def test_loads_from_environment(self, monkeypatch):
        """从环境变量正确加载"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v3")
        monkeypatch.setenv("AGENT_MAX_ITERATIONS", "5")
        monkeypatch.setenv("AGENT_TRACE_ENABLED", "false")

        cfg = Config.from_env(env_file=None)
        assert cfg.llm_api_key == "sk-test"
        assert cfg.llm_model == "deepseek-v3"
        assert cfg.max_iterations == 5
        assert cfg.trace_enabled is False

    def test_loads_from_env_file(self, monkeypatch):
        """从 .env 文件正确加载"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env-file")
        monkeypatch.setenv("AGENT_MAX_ITERATIONS", "7")

        # 使用不存在的 .env 文件路径（仅依赖环境变量）
        cfg = Config.from_env(env_file="/nonexistent/.env")
        assert cfg.llm_api_key == "sk-from-env-file"
        assert cfg.max_iterations == 7

    def test_env_file_missing_no_error(self, monkeypatch):
        """.env 文件不存在时不报错（回退到环境变量或默认值）"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        cfg = Config.from_env(env_file="/no/such/file.env")
        assert cfg.llm_api_key == "sk-test"

    def test_env_file_parsing(self, monkeypatch):
        """测试 .env 文件解析"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
        monkeypatch.delenv("AGENT_MAX_ITERATIONS", raising=False)

        # 创建临时 .env 文件
        env_content = """DEEPSEEK_API_KEY=sk-temp-key
DEEPSEEK_MODEL=custom-model
# This is a comment
AGENT_MAX_ITERATIONS=3
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(env_content)
            tmp_path = f.name

        try:
            cfg = Config.from_env(env_file=tmp_path)
            assert cfg.llm_api_key == "sk-temp-key"
            assert cfg.llm_model == "custom-model"
            assert cfg.max_iterations == 3
        finally:
            os.unlink(tmp_path)
            # 清理写入 os.environ 的变量，避免污染后续测试
            for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "AGENT_MAX_ITERATIONS"):
                os.environ.pop(key, None)

    def test_env_var_overrides_file(self, monkeypatch):
        """环境变量优先级高于 .env 文件"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        monkeypatch.setenv("AGENT_MAX_ITERATIONS", "20")
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)  # 清理可能残留

        env_content = "DEEPSEEK_API_KEY=sk-from-file\nAGENT_MAX_ITERATIONS=5\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write(env_content)
            tmp_path = f.name

        try:
            cfg = Config.from_env(env_file=tmp_path)
            # 环境变量优先
            assert cfg.llm_api_key == "sk-from-env"
            assert cfg.max_iterations == 20
        finally:
            os.unlink(tmp_path)
            os.environ.pop("DEEPSEEK_MODEL", None)


# ============================================================
# Phase 4 — YAML 配置支持
# ============================================================


class TestFromYaml:
    """Config.from_yaml() 测试"""

    def test_loads_llm_section(self):
        yaml_content = """\
llm:
  model: "deepseek-v3"
  timeout: 120
  max_retries: 5
"""
        path = _write_temp_yaml(yaml_content)
        try:
            cfg = Config.from_yaml(path)
            assert cfg.llm_model == "deepseek-v3"
            assert cfg.llm_timeout == 120
            assert cfg.llm_max_retries == 5
            # 未设置的字段保持默认值
            assert cfg.max_iterations == 10
        finally:
            os.unlink(path)

    def test_loads_all_sections(self):
        yaml_content = """\
llm:
  api_key: "sk-yaml-key"
  model: "custom-model"
loop:
  max_iterations: 5
tool:
  timeout: 60
paths:
  sessions_dir: "./yaml-sessions"
trace:
  enabled: false
compression:
  threshold: 0.5
  keep_recent: 3
"""
        path = _write_temp_yaml(yaml_content)
        try:
            cfg = Config.from_yaml(path)
            assert cfg.llm_api_key == "sk-yaml-key"
            assert cfg.llm_model == "custom-model"
            assert cfg.max_iterations == 5
            assert cfg.tool_timeout == 60
            assert cfg.sessions_dir == "./yaml-sessions"
            assert cfg.trace_enabled is False
            assert cfg.compression_threshold == 0.5
            assert cfg.compression_keep_recent == 3
        finally:
            os.unlink(path)

    def test_missing_file_returns_defaults(self):
        """YAML 文件不存在时不报错，返回全默认值"""
        cfg = Config.from_yaml("/no/such/config.yaml")
        assert cfg.llm_model == "deepseek-chat"
        assert cfg.max_iterations == 10

    def test_empty_file_returns_defaults(self):
        """空 YAML 文件返回全默认值"""
        path = _write_temp_yaml("")
        try:
            cfg = Config.from_yaml(path)
            assert cfg.llm_model == "deepseek-chat"
        finally:
            os.unlink(path)

    def test_partial_override(self):
        """YAML 中只覆盖部分字段，其余保持默认"""
        yaml_content = "llm:\n  model: \"partial-model\"\n"
        path = _write_temp_yaml(yaml_content)
        try:
            cfg = Config.from_yaml(path)
            assert cfg.llm_model == "partial-model"
            assert cfg.llm_base_url == "https://api.deepseek.com/v1"  # 默认值
            assert cfg.max_iterations == 10  # 默认值
        finally:
            os.unlink(path)


class TestFromEnvWithYaml:
    """from_env() 与 YAML 优先级测试"""

    def test_env_overrides_yaml(self, monkeypatch):
        """环境变量 > YAML"""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
        monkeypatch.setenv("DEEPSEEK_MODEL", "env-model")
        monkeypatch.setenv("AGENT_MAX_ITERATIONS", "42")

        yaml_content = """\
llm:
  model: "yaml-model"
loop:
  max_iterations: 5
"""
        yaml_path = _write_temp_yaml(yaml_content)
        try:
            cfg = Config.from_env(env_file=None, yaml_file=yaml_path)
            assert cfg.llm_api_key == "sk-from-env"
            assert cfg.llm_model == "env-model"  # env > yaml
            assert cfg.max_iterations == 42  # env > yaml
        finally:
            os.unlink(yaml_path)

    def test_env_file_overrides_yaml(self, monkeypatch):
        """.env > YAML"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
        monkeypatch.delenv("AGENT_MAX_ITERATIONS", raising=False)

        yaml_content = """\
llm:
  model: "yaml-model"
loop:
  max_iterations: 5
"""
        yaml_path = _write_temp_yaml(yaml_content)

        env_content = "DEEPSEEK_API_KEY=sk-from-dotenv\nDEEPSEEK_MODEL=dotenv-model\n"
        env_f = tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False)
        env_f.write(env_content)
        env_f.close()

        try:
            cfg = Config.from_env(env_file=env_f.name, yaml_file=yaml_path)
            assert cfg.llm_api_key == "sk-from-dotenv"
            assert cfg.llm_model == "dotenv-model"  # .env > yaml
            assert cfg.max_iterations == 5  # .env 没设，fallback to yaml
        finally:
            os.unlink(yaml_path)
            os.unlink(env_f.name)

    def test_yaml_base_when_no_env(self, monkeypatch):
        """env 和 .env 都没有时，YAML 作为 base"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
        monkeypatch.delenv("AGENT_MAX_ITERATIONS", raising=False)

        yaml_content = """\
llm:
  api_key: "sk-yaml-only"
  model: "yaml-only-model"
loop:
  max_iterations: 8
"""
        yaml_path = _write_temp_yaml(yaml_content)
        try:
            cfg = Config.from_env(env_file=None, yaml_file=yaml_path)
            assert cfg.llm_api_key == "sk-yaml-only"
            assert cfg.llm_model == "yaml-only-model"
            assert cfg.max_iterations == 8
        finally:
            os.unlink(yaml_path)
