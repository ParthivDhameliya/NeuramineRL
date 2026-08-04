"""A deterministic mock e-commerce orders API with five undocumented quirks.

The "docs" given to the agent (DOCS below) deliberately omit the rules:

1. ``ship_date`` must be ISO YYYY-MM-DD.
2. ``amount`` must be an integer number of cents.
3. Every POST needs an ``Idempotency-Key`` header.
4. ``country`` must be an uppercase ISO-2 code.
5. Some orders hit a 409 conflict; the fix is to retry with a NEW
   idempotency key, never to resubmit the same one.

Pure Python, no network — perfect for CI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

DOCS = """\
Orders API: POST /orders
Required payload fields:
  item       - product name
  quantity   - number of units
  amount     - the total amount for the order
  country    - destination country
  ship_date  - requested ship date
Respond to errors by correcting your request."""

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_COUNTRIES = {"US", "CA", "GB", "DE", "FR", "JP", "IN", "AU", "BR", "MX"}


@dataclass
class ApiResult:
    status: int
    body: str

    @property
    def ok(self) -> bool:
        return self.status == 201


@dataclass
class OrdersAPI:
    """One instance per episode: burned idempotency keys reset each time."""

    _burned_keys: dict[str, set[str]] = field(default_factory=dict)

    def place_order(
        self, payload: dict[str, Any], headers: dict[str, str], *, flaky_id: str | None = None
    ) -> ApiResult:
        key = next((v for k, v in headers.items() if k.lower() == "idempotency-key"), None)
        if not key:
            return ApiResult(400, "request rejected: missing required header 'Idempotency-Key'")

        if flaky_id is not None:
            burned = self._burned_keys.setdefault(flaky_id, set())
            if key in burned:
                return ApiResult(
                    409,
                    "conflict: an order with this Idempotency-Key may already exist; "
                    "do not resubmit the same key",
                )
            if not burned:  # first attempt for this order always conflicts
                burned.add(key)
                return ApiResult(
                    409,
                    "conflict: an order with this Idempotency-Key may already exist; "
                    "do not resubmit the same key",
                )

        ship_date = payload.get("ship_date")
        if not isinstance(ship_date, str) or not _ISO_DATE.match(ship_date):
            return ApiResult(400, "invalid value for field 'ship_date'")

        amount = payload.get("amount")
        if not isinstance(amount, int) or isinstance(amount, bool):
            return ApiResult(400, "field 'amount' has invalid type")

        country = payload.get("country")
        if not isinstance(country, str) or country not in _COUNTRIES:
            return ApiResult(400, "invalid value for field 'country'")

        if not payload.get("item") or not isinstance(payload.get("quantity"), int):
            return ApiResult(400, "missing or invalid 'item'/'quantity'")

        return ApiResult(201, f"created order for {payload['quantity']}x {payload['item']}")


@dataclass(frozen=True)
class OrderTask:
    """A natural-language order the agent must place."""

    id: str
    item: str
    quantity: int
    unit_price_dollars: float
    country_name: str
    country_code: str
    ship_date_text: str
    ship_date_iso: str
    flaky: bool = False

    def describe(self) -> str:
        return (
            f"Place an order: {self.quantity}x {self.item} at "
            f"${self.unit_price_dollars:.2f} each, shipping to {self.country_name}, "
            f"requested ship date {self.ship_date_text}."
        )

    @property
    def amount_cents(self) -> int:
        return round(self.quantity * self.unit_price_dollars * 100)


TASKS = [
    OrderTask("t1", "blue widget", 3, 4.99, "Germany", "DE", "March 5, 2026", "2026-03-05"),
    OrderTask("t2", "red gadget", 1, 19.50, "Canada", "CA", "April 12, 2026", "2026-04-12"),
    OrderTask("t3", "green gizmo", 7, 2.25, "Japan", "JP", "May 1, 2026", "2026-05-01"),
    OrderTask("t4", "solar charger", 2, 34.00, "Brazil", "BR", "June 20, 2026", "2026-06-20"),
    OrderTask(
        "t5",
        "usb cable",
        10,
        1.99,
        "United States",
        "US",
        "July 4, 2026",
        "2026-07-04",
        flaky=True,
    ),
]
