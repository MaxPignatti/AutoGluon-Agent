"""Exception hierarchy. The CLI maps AgaError to exit 1 and BlockedError to exit 2."""


class AgaError(Exception):
    """Base error for expected failures (exit code 1)."""


class BlockedError(AgaError):
    """A guardrail blocked the operation (exit code 2)."""


class HoldoutReusedError(BlockedError):
    """The holdout set was already used and cannot be scored again."""


class FeatureModuleError(AgaError):
    """A user feature module failed to load or run."""


class ContractError(FeatureModuleError):
    """A feature module violated the feature contract."""
