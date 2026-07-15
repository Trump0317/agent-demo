"""MVT 1.3 / Phase 4 — 配置管理

Config 数据类，支持从 YAML 文件、.env 文件和环境变量加载配置。

优先级：环境变量 > .env 文件 > config.yaml > 默认值
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from src.agent.exceptions import ConfigError

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # ── LLM ──
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_timeout: int = 60
    llm_max_retries: int = 2

    # ── Agent Loop ──
    max_iterations: int = 10
    max_context_tokens: int = 64000

    # ── Tool ──
    tool_timeout: int = 30

    # ── Paths ──
    sessions_dir: str = "./sessions"
    traces_dir: str = "./traces"

    # ── Trace ──
    trace_enabled: bool = True
    trace_content_max_length: int = 500

    # ── Compression ──
    compression_threshold: float = 0.8
    compression_keep_recent: int = 5

    # ────────────────────────────────────────
    # 工厂方法
    # ────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str = "config/config.yaml") -> "Config":
        """从 YAML 文件加载配置

        Args:
            path: YAML 配置文件路径

        Returns:
            Config 实例（只包含 YAML 中明确设置的字段，其余用默认值）

        Raises:
            ConfigError: YAML 文件存在但解析失败时抛出
            FileNotFoundError: YAML 文件不存在时由底层函数抛出
        """
        yaml_data = cls._load_yaml_file(path)
        if not yaml_data:
            return cls()

        kwargs: dict = {}

        # ── llm 段 ──
        llm = yaml_data.get("llm", {})
        if isinstance(llm, dict):
            if "api_key" in llm and llm["api_key"]:
                kwargs["llm_api_key"] = str(llm["api_key"])
            if "base_url" in llm:
                kwargs["llm_base_url"] = str(llm["base_url"])
            if "model" in llm:
                kwargs["llm_model"] = str(llm["model"])
            if "timeout" in llm:
                kwargs["llm_timeout"] = int(llm["timeout"])
            if "max_retries" in llm:
                kwargs["llm_max_retries"] = int(llm["max_retries"])

        # ── loop 段 ──
        loop = yaml_data.get("loop", {})
        if isinstance(loop, dict):
            if "max_iterations" in loop:
                kwargs["max_iterations"] = int(loop["max_iterations"])
            if "max_context_tokens" in loop:
                kwargs["max_context_tokens"] = int(loop["max_context_tokens"])

        # ── tool 段 ──
        tool = yaml_data.get("tool", {})
        if isinstance(tool, dict):
            if "timeout" in tool:
                kwargs["tool_timeout"] = int(tool["timeout"])

        # ── paths 段 ──
        paths = yaml_data.get("paths", {})
        if isinstance(paths, dict):
            if "sessions_dir" in paths:
                kwargs["sessions_dir"] = str(paths["sessions_dir"])
            if "traces_dir" in paths:
                kwargs["traces_dir"] = str(paths["traces_dir"])

        # ── trace 段 ──
        trace = yaml_data.get("trace", {})
        if isinstance(trace, dict):
            if "enabled" in trace:
                kwargs["trace_enabled"] = bool(trace["enabled"])
            if "content_max_length" in trace:
                kwargs["trace_content_max_length"] = int(trace["content_max_length"])

        # ── compression 段 ──
        compression = yaml_data.get("compression", {})
        if isinstance(compression, dict):
            if "threshold" in compression:
                kwargs["compression_threshold"] = float(compression["threshold"])
            if "keep_recent" in compression:
                kwargs["compression_keep_recent"] = int(compression["keep_recent"])

        return cls(**kwargs)

    @classmethod
    def from_env(
        cls,
        env_file: str | None = ".env",
        yaml_file: str | None = "config/config.yaml",
    ) -> "Config":
        """从 YAML 文件、.env 文件和环境变量加载配置。

        优先级（从低到高）：
        1. 默认值
        2. config.yaml
        3. .env 文件
        4. 环境变量（最高）

        Args:
            env_file: .env 文件路径，None 表示不加载
            yaml_file: YAML 配置文件路径，None 表示不加载

        Returns:
            Config 实例

        Raises:
            ConfigError: 缺少必填配置项 DEEPSEEK_API_KEY 时抛出
        """
        # 层 1: YAML 文件
        base = cls()
        if yaml_file is not None and os.path.isfile(yaml_file):
            try:
                base = cls.from_yaml(yaml_file)
            except Exception as e:
                logger.warning(f"Failed to load YAML config '{yaml_file}': {e}")

        # 层 2: .env 文件（写入 environ，不覆盖已有变量）
        if env_file is not None:
            cls._load_env_file(env_file)

        # 层 3: 环境变量（覆盖 YAML 值）
        llm_api_key = os.getenv("DEEPSEEK_API_KEY", base.llm_api_key)
        if not llm_api_key:
            raise ConfigError(
                "Missing required config: DEEPSEEK_API_KEY. "
                "Set it via environment variable, .env file, or config/config.yaml."
            )

        return cls(
            llm_api_key=llm_api_key,
            llm_base_url=os.getenv("DEEPSEEK_BASE_URL", base.llm_base_url),
            llm_model=os.getenv("DEEPSEEK_MODEL", base.llm_model),
            llm_timeout=int(os.getenv("AGENT_LLM_TIMEOUT", str(base.llm_timeout))),
            llm_max_retries=int(os.getenv("AGENT_LLM_MAX_RETRIES", str(base.llm_max_retries))),
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", str(base.max_iterations))),
            max_context_tokens=int(os.getenv("AGENT_MAX_CONTEXT_TOKENS", str(base.max_context_tokens))),
            tool_timeout=int(os.getenv("AGENT_TOOL_TIMEOUT", str(base.tool_timeout))),
            sessions_dir=os.getenv("AGENT_SESSIONS_DIR", base.sessions_dir),
            traces_dir=os.getenv("AGENT_TRACES_DIR", base.traces_dir),
            trace_enabled=os.getenv("AGENT_TRACE_ENABLED", str(base.trace_enabled)).lower() != "false",
            trace_content_max_length=int(
                os.getenv("AGENT_TRACE_CONTENT_MAX_LENGTH", str(base.trace_content_max_length))
            ),
            compression_threshold=float(
                os.getenv("AGENT_COMPRESSION_THRESHOLD", str(base.compression_threshold))
            ),
            compression_keep_recent=int(
                os.getenv("AGENT_COMPRESSION_KEEP_RECENT", str(base.compression_keep_recent))
            ),
        )

    # ────────────────────────────────────────
    # 内部辅助
    # ────────────────────────────────────────

    @staticmethod
    def _load_env_file(path: str) -> None:
        """解析 .env 文件，写入 os.environ（不覆盖已有环境变量）"""
        if not os.path.isfile(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # 去掉引号
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                if key not in os.environ:
                    os.environ[key] = value

    @staticmethod
    def _load_yaml_file(path: str) -> dict:
        """解析 YAML 文件，返回 dict。文件不存在或 PyYAML 不可用时返回空 dict。"""
        if not os.path.isfile(path):
            return {}

        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed, cannot load YAML config")
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            return {}
        if not isinstance(data, dict):
            logger.warning(f"YAML config '{path}' is not a dict, ignoring")
            return {}
        return data
