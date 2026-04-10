import logging

logger = logging.getLogger(__name__)


def patch_qwen2_rope_theta_compat() -> None:
    """Patch Qwen2Config to expose a compatible ``rope_theta`` attribute.

    Some bridge/model code accesses ``config.rope_theta`` directly, while newer
    Transformers variants may only store this value inside ``rope_parameters``
    (or ``rope_scaling``). This patch adds a property fallback on class level.
    """

    try:
        from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    except Exception:
        return

    # If class already provides rope_theta, keep native behavior.
    if hasattr(Qwen2Config, "rope_theta"):
        return

    def _rope_theta(self):
        rope_params = getattr(self, "rope_parameters", None)
        if not isinstance(rope_params, dict):
            rope_params = getattr(self, "rope_scaling", None)
        if isinstance(rope_params, dict):
            value = rope_params.get("rope_theta")
            if value is not None:
                return value
        # Qwen2 default in most configs.
        return 1000000

    try:
        Qwen2Config.rope_theta = property(_rope_theta)  # type: ignore[assignment]
        logger.info("[HFCompat] patched Qwen2Config.rope_theta fallback property.")
    except Exception:
        logger.exception("[HFCompat] failed to patch Qwen2Config.rope_theta compatibility.")
