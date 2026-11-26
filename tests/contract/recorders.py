"""Recorders that let scenarios assert parallelism, order and progress without sleeps."""

import asyncio
import itertools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

BOUND = 1.0


class Intervals:
    def __init__(self) -> None:
        self._clock = itertools.count()
        self.starts: dict[str, int] = {}
        self.ends: dict[str, int] = {}

    def start(self, name: str) -> None:
        if name in self.starts:
            raise ValueError(f"step {name!r} already started")
        self.starts[name] = next(self._clock)

    def end(self, name: str) -> None:
        if name not in self.starts:
            raise ValueError(f"step {name!r} ended before it started")
        if name in self.ends:
            raise ValueError(f"step {name!r} already ended")
        self.ends[name] = next(self._clock)

    @asynccontextmanager
    async def span(self, name: str) -> AsyncIterator[None]:
        self.start(name)
        try:
            yield
        finally:
            self.end(name)

    def _finished(self, names: tuple[str, ...]) -> None:
        if len(set(names)) < 2 or len(set(names)) != len(names):
            raise ValueError(f"name at least two distinct steps, got {list(names)}")
        for name in names:
            if name not in self.starts:
                raise AssertionError(f"step {name!r} was never started")
            if name not in self.ends:
                raise AssertionError(f"step {name!r} never ended")

    def assert_overlap(self, *names: str) -> None:
        self._finished(names)
        if max(self.starts[name] for name in names) > min(self.ends[name] for name in names):
            raise AssertionError(f"steps {list(names)} never ran at one moment")

    def assert_order(self, *names: str) -> None:
        self._finished(names)
        for before, after in itertools.pairwise(names):
            if self.starts[after] < self.ends[before]:
                raise AssertionError(f"step {after!r} started before step {before!r} ended")


def _checked(bound: float) -> float:
    if bound <= 0:
        raise ValueError(f"a wait bound must be positive, got {bound}")
    return bound


class Gate:
    def __init__(self, name: str, *, bound: float = BOUND) -> None:
        self.name = name
        self.bound = _checked(bound)
        self._event = asyncio.Event()

    def open(self) -> None:
        self._event.set()

    async def wait(self) -> None:
        try:
            async with asyncio.timeout(self.bound):
                await self._event.wait()
        except TimeoutError:
            raise AssertionError(f"gate {self.name!r} did not open within {self.bound}s") from None


class Meeting:
    def __init__(self, name: str, *, parties: int, bound: float = BOUND) -> None:
        if parties < 2:
            raise ValueError(f"a meeting needs at least two parties, got {parties}")
        self.name = name
        self.parties = parties
        self.bound = _checked(bound)
        self.arrived = 0
        self.failure: str | None = None
        self._event = asyncio.Event()

    async def arrive(self) -> None:
        if self.failure is not None:
            raise AssertionError(f"meeting {self.name!r} already failed: {self.failure}")
        if self.arrived == self.parties:
            raise ValueError(f"meeting {self.name!r} already has its {self.parties} parties")
        self.arrived += 1
        if self.arrived == self.parties:
            self._event.set()
        try:
            async with asyncio.timeout(self.bound):
                await self._event.wait()
        except TimeoutError:
            self.failure = f"got {self.arrived} of {self.parties} parties within {self.bound}s"
            raise AssertionError(f"meeting {self.name!r} {self.failure}") from None
        except asyncio.CancelledError:
            self.failure = f"a party was cancelled after {self.arrived} of {self.parties} arrived"
            raise


class Progress:
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    async def publish(self, step: str, percent: int) -> None:
        if type(percent) is not int or not 0 <= percent <= 100:
            raise ValueError(f"progress is a whole percent from 0 to 100, got {percent!r}")
        self.events.append((step, percent))
