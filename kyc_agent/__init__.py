"""AGY-KYC Checker Agent Package."""

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from .agent import root_agent

__all__ = ["root_agent"]

