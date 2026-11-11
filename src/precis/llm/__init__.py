"""Model access: the gateway every request goes through, and its test double."""

from precis.llm.base import ModelGateway, Reply
from precis.llm.client import AsyncAnthropicLike, make_client
from precis.llm.clock import Clock, ManualClock, RealClock, ScaledClock
from precis.llm.errors import BudgetExceeded, GatewayError, RefusalError
from precis.llm.gateway import Gateway, GatewayStats
from precis.llm.pricing import UnpricedModelError

__all__ = [
    "AsyncAnthropicLike",
    "BudgetExceeded",
    "Clock",
    "Gateway",
    "GatewayError",
    "GatewayStats",
    "ManualClock",
    "ModelGateway",
    "RealClock",
    "RefusalError",
    "Reply",
    "ScaledClock",
    "UnpricedModelError",
    "make_client",
]
