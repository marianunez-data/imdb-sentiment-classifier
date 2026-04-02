"""Structured JSON logging with structlog.

All modules should use get_logger() instead of print().
"""

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON output and standard processors.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ],
            ),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper()),
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


# Initialize logging on module import
setup_logging()


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named structured logger instance.

    Args:
        name: Logger name, typically the module name.

    Returns:
        Bound structlog logger.
    """
    return structlog.get_logger(name)
