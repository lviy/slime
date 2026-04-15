import logging

logger = logging.getLogger(__name__)


def patch_qwen2_rope_theta_compat() -> None:
    """Patch Qwen2/Qwen3 config classes to expose a compatible ``rope_theta`` attribute.

    Some bridge/model code accesses ``config.rope_theta`` directly, while newer
    Transformers variants may only store this value inside ``rope_parameters``
    (or ``rope_scaling``). This patch adds a property fallback on class level.
    """

    def _patch_config_class(config_cls, config_name: str) -> None:
        if hasattr(config_cls, "rope_theta"):
            return

        def _rope_theta(self):
            rope_params = getattr(self, "rope_parameters", None)
            if not isinstance(rope_params, dict):
                rope_params = getattr(self, "rope_scaling", None)
            if isinstance(rope_params, dict):
                value = rope_params.get("rope_theta")
                if value is not None:
                    return value
            # Qwen2/Qwen3 default in most configs.
            return 1000000

        try:
            config_cls.rope_theta = property(_rope_theta)  # type: ignore[assignment]
            logger.info(f"[HFCompat] patched {config_name}.rope_theta fallback property.")
        except Exception:
            logger.exception(f"[HFCompat] failed to patch {config_name}.rope_theta compatibility.")

    try:
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    except Exception:
        Qwen2Config = None

    if Qwen2Config is not None:
        _patch_config_class(Qwen2Config, "Qwen2Config")

    try:
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    except Exception:
        Qwen3Config = None

    if Qwen3Config is not None:
        _patch_config_class(Qwen3Config, "Qwen3Config")
