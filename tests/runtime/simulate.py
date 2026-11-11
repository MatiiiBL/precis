"""Closed-loop traffic simulations of the gateway against ``FakeAnthropic``.

The convergence tests use ``simulate``; running this module measures the
numbers quoted in docs/runtime.md and draws docs/img/runtime-aimd.png:

    uv run python -m tests.runtime.simulate
"""

from __future__ import annotations

import asyncio
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from precis.llm import Gateway, GatewayError, ScaledClock
from precis.llm.fake import FakeAnthropic, FakeReply, Faults, LatencyModel, ServerLimits
from precis.llm.limiter import AIMDWindow, Outcome
from precis.store import Store

from .helpers import config

PROMPT_CHARS = 4000  # ~1,000 input tokens per request
OUTPUT_TEXT = "token " * 200  # ~300 output tokens per reply


@dataclass
class Sample:
    t: float
    in_flight: int
    limit: int
    queued: int
    rpm_limit: float | None


@dataclass
class SimResult:
    duration_s: float
    fake: FakeAnthropic
    gateway: Gateway
    samples: list[Sample] = field(default_factory=list)
    failures: int = 0

    def ok_times(self) -> list[float]:
        return [e.t for e in self.fake.events if e.outcome == "ok"]

    def rate_per_min(self, start: float, end: float, outcome: str = "ok") -> float:
        n = sum(start <= e.t < end for e in self.fake.events if e.outcome == outcome)
        return n * 60 / (end - start)

    def tokens_per_min(self, start: float, end: float) -> tuple[float, float]:
        events = [e for e in self.fake.events if e.outcome == "ok" and start <= e.t < end]
        scale = 60 / (end - start)
        return (
            sum(e.input_tokens for e in events) * scale,
            sum(e.output_tokens for e in events) * scale,
        )

    def count(self, outcome: str, start: float = 0.0, end: float = math.inf) -> int:
        return sum(start <= e.t < end for e in self.fake.events if e.outcome == outcome)


class FixedWindow(AIMDWindow):
    """A concurrency limit that ignores overload: the baseline AIMD is compared with."""

    def release(self, epoch: int, outcome: Outcome) -> None:
        self.in_flight -= 1


def _reply(params: dict[str, Any]) -> FakeReply:
    return FakeReply.text(OUTPUT_TEXT)


async def simulate(
    *,
    duration_s: float,
    clients: int,
    speed: float,
    limits: ServerLimits | None = None,
    faults: Faults | None = None,
    change_at: float | None = None,
    new_limits: ServerLimits | None = None,
    gateway_limits: dict[str, Any] | None = None,
    fixed_concurrency: int | None = None,
    seed: int = 0,
) -> SimResult:
    """Run ``clients`` agents that send requests back to back for ``duration_s``."""
    clock = ScaledClock(speed)
    fake = FakeAnthropic(
        [_reply],
        clock=clock,
        seed=seed,
        latency=LatencyModel(first_token_s=1.5, seconds_per_output_token=0.01, jitter=0.3),
        limits=limits,
        faults=faults,
    )
    cfg = config(
        limits={"max_attempts": 20, **(gateway_limits or {})},
        pipeline={"use_cache": False, "fallbacks": False},
    )
    gw = Gateway(fake, Store(":memory:"), run_id=f"sim-{seed}", config=cfg, clock=clock)
    if fixed_concurrency is not None:
        gw._admission.window = FixedWindow(fixed_concurrency, fixed_concurrency)
    result = SimResult(duration_s, fake, gw)
    counter = 0

    async def client() -> None:
        nonlocal counter
        while clock.now() < duration_s:
            counter += 1
            params = {
                "model": "claude-opus-5-5",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": f"{counter} " + "x" * PROMPT_CHARS}],
            }
            try:
                await gw.create(params, stage="read")
            except GatewayError:
                result.failures += 1

    async def sampler() -> None:
        changed = False
        while clock.now() < duration_s:
            if change_at is not None and not changed and clock.now() >= change_at:
                fake.set_limits(new_limits)
                changed = True
            stats = gw.stats()
            result.samples.append(
                Sample(
                    clock.now(),
                    stats.in_flight,
                    stats.concurrency_limit,
                    stats.queued,
                    stats.rpm_limit,
                )
            )
            await clock.sleep(1.0)

    await asyncio.gather(sampler(), *(client() for _ in range(clients)))
    return result


def _learned_after(result: SimResult, after: float, rpm: float) -> float:
    """Seconds after ``after`` until the gateway's request limit equals ``rpm``."""
    times = [s.t for s in result.samples if s.t >= after and s.rpm_limit == rpm]
    return times[0] - after if times else math.inf


def _windows(result: SimResult, start: float, end: float, width: float) -> list[float]:
    return [result.rate_per_min(t, t + width) for t in range(int(start), int(end - width) + 1)]


async def _measure() -> None:
    print("Rate-limited run: server 600 RPM / 1.2M ITPM / 240K OTPM, limits learned")
    limits = ServerLimits(rpm=600, itpm=1_200_000, otpm=240_000)
    halved = ServerLimits(rpm=300, itpm=1_200_000, otpm=240_000)
    r = await simulate(
        duration_s=900,
        clients=64,
        speed=100,
        limits=limits,
        change_at=450,
        new_limits=halved,
        gateway_limits={"max_concurrency": 64},
    )
    total = len(r.fake.events)
    ramp = next(s.t for s in r.samples if s.limit >= 45)
    print(f"  concurrency reached 45 (enough for 600 RPM at ~4.5 s latency) at {ramp:.0f}s")
    for start, end in ((60.0, 450.0), (240.0, 450.0)):
        inp, out = r.tokens_per_min(start, end)
        print(
            f"  {start:.0f}-{end:.0f}s: {r.rate_per_min(start, end):.0f} requests/min of 600, "
            f"{inp:,.0f} input tok/min, {out:,.0f} output tok/min"
        )
    print(f"  429s: {r.count('429')} of {total} responses ({100 * r.count('429') / total:.2f}%)")
    print(f"  new 300 RPM limit learned {_learned_after(r, 450, 300):.1f}s after the change")
    after = _windows(r, 455, 900, 30)
    print(
        f"  30-s windows after the change: max {max(after):.0f} RPM, "
        f"mean {sum(after) / len(after):.0f} RPM; 429s after the change: {r.count('429', 450)}"
    )
    print(f"  failures: {r.failures}")

    print("The same run without slow start (one slot per window from 8)")
    r = await simulate(
        duration_s=450,
        clients=64,
        speed=100,
        limits=limits,
        gateway_limits={"max_concurrency": 64, "slow_start": False},
    )
    ramp = next(s.t for s in r.samples if s.limit >= 45)
    print(f"  concurrency reached 45 at {ramp:.0f}s")
    print(f"  60-450s: {r.rate_per_min(60.0, 450.0):.0f} requests/min of 600")
    print(f"  429s: {r.count('429')}")

    print("Overload run: 529 above 24 concurrent requests, no rate limits")
    for name, fixed in (("AIMD", None), ("fixed 64", 64)):
        r = await simulate(
            duration_s=600,
            clients=64,
            speed=100,
            faults=Faults(overload_above=24),
            gateway_limits={"max_concurrency": 64, "initial_concurrency": 8},
            fixed_concurrency=fixed,
        )
        ok = r.count("ok", 60, 600)
        overloaded = r.count("529", 60, 600)
        print(
            f"  {name}: {r.rate_per_min(60, 600):.0f} ok/min, "
            f"{100 * overloaded / max(1, ok + overloaded):.1f}% of responses 529, "
            f"mean in flight {sum(s.in_flight for s in r.samples) / len(r.samples):.1f}"
        )
        if fixed is None:
            _plot(r, Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/img/runtime-aimd.png"))


def _plot(result: SimResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    t = [s.t / 60 for s in result.samples]
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.axhline(24, color=muted, ls=(0, (4, 3)), lw=1, label="server capacity (529 above)")
    ax.plot(t, [s.limit for s in result.samples], color="#2a78d6", lw=2, label="AIMD limit")
    ax.plot(t, [s.in_flight for s in result.samples], color="#eb6834", lw=1.2, label="in flight")
    ax.set_xlabel("simulated minutes", color=muted)
    ax.set_ylabel("concurrent requests", color=muted)
    ax.set_ylim(0, 40)
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted, labelcolor=muted)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=8, ncols=3)
    for text in legend.get_texts():
        text.set_color(ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"  figure written to {path}")


if __name__ == "__main__":
    asyncio.run(_measure())
