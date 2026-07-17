"""SolFoodHook end-to-end transport behavior (synthetic, no network)."""

import asyncio
import base64
import hashlib
import json
import stat
import os
from pathlib import Path

import pytest

from gateway.topic_hooks import HookDecision, MediaDescriptor
from gateway.topic_routing import RouteOrigin, TopicRoute
from plugins.sol_food.health_client import HealthClientError
from plugins.sol_food.hook import SolFoodHook
from plugins.sol_food.legacy_guard import LegacyHelperPresent
from plugins.sol_food.limits import (
    FOOD_PARSE_DEADLINE_SECONDS,
    FOOD_TEXT_MAX_CHARS,
)
from plugins.sol_food.proposal import ACTION_CANCEL, ACTION_CONFIRM, Candidate

SOL = TopicRoute("208214988", 1, "sol")


def origin(update_id=1000, message_id=42, thread_id=1):
    return RouteOrigin("999000", "208214988", thread_id, update_id, message_id)


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


class Replies:
    def __init__(self):
        self.messages = []

    async def __call__(self, text: str) -> None:
        self.messages.append(text)


class FakeHealthClient:
    """Records commit calls; programmable outcome."""

    def __init__(self):
        self.calls = []
        self.error = None
        self.replayed = False

    def commit(self, envelope):
        self.calls.append(envelope.request_bytes)
        if self.error is not None:
            raise self.error
        receipt = {
            "schema": "health.food_commit_receipt.v1",
            "mutation_id": envelope.mutation_id,
            "entry_id": envelope.entry_id,
            "operation": envelope.operation,
            "request_sha256": envelope.request_sha256,
            "status": "applied",
            "same_transaction_readback_match": True,
        }
        from plugins.sol_food.health_client import VerifiedCommit

        receipt_bytes = json.dumps(receipt, sort_keys=True).encode()
        return VerifiedCommit(
            receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
            replayed=self.replayed,
            receipt=receipt,
        )


def sample_candidates(n=2):
    return [
        Candidate(
            label=f"synthetic option {i}",
            items=({"plant_key": f"synthetic_item_{i}", "is_plant": True},),
        )
        for i in range(n)
    ]


async def default_parser(text, image_path):
    return sample_candidates()


@pytest.fixture()
def clock():
    return Clock()


@pytest.fixture()
def health():
    return FakeHealthClient()


@pytest.fixture()
def hook(tmp_path, clock, health):
    return SolFoodHook(
        state_dir=tmp_path / "state",
        hermes_home=tmp_path / "hermes-home",
        health_client=health,
        parser=default_parser,
        clock=clock,
    )


class TestLegacyGuard:
    def test_refuses_activation_with_helper_present(self, tmp_path, health):
        home = tmp_path / "hermes-home"
        (home / "scripts").mkdir(parents=True)
        (home / "scripts" / "food_log_commit.py").write_text("# legacy")
        with pytest.raises(LegacyHelperPresent):
            SolFoodHook(
                state_dir=tmp_path / "state",
                hermes_home=home,
                health_client=health,
                parser=default_parser,
            )

    def test_refuses_nudge_helper_too(self, tmp_path, health):
        home = tmp_path / "hermes-home"
        (home / "scripts").mkdir(parents=True)
        (home / "scripts" / "food_nudge.py").write_text("# legacy")
        with pytest.raises(LegacyHelperPresent):
            SolFoodHook(
                state_dir=tmp_path / "state",
                hermes_home=home,
                health_client=health,
                parser=default_parser,
            )


class TestConversation:
    @pytest.mark.asyncio
    async def test_ordinary_text_continues(self, hook):
        replies = Replies()
        decision = await hook.on_message(SOL, origin(), "hello sol", replies)
        assert decision is HookDecision.CONTINUE
        assert replies.messages == []


class TestTextProposal:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "length,ok",
        [
            (FOOD_TEXT_MAX_CHARS - 1, True),
            (FOOD_TEXT_MAX_CHARS, True),
            (FOOD_TEXT_MAX_CHARS + 1, False),
        ],
    )
    async def test_text_length_boundary(self, hook, length, ok):
        replies = Replies()
        proposal_id = await hook.propose_from_text(origin(), "x" * length, replies)
        if ok:
            assert proposal_id is not None
        else:
            assert proposal_id is None

    @pytest.mark.asyncio
    async def test_no_write_before_confirm(self, hook, health):
        replies = Replies()
        proposal_id = await hook.propose_from_text(origin(), "synthetic meal", replies)
        assert proposal_id is not None
        assert health.calls == []  # candidates never write


class TestParserCeilings:
    @pytest.mark.asyncio
    async def test_two_attempts_max(self, tmp_path, clock, health):
        attempts = []

        async def flaky(text, image_path):
            attempts.append(1)
            raise RuntimeError("nope")

        hook = SolFoodHook(
            state_dir=tmp_path / "s2",
            hermes_home=tmp_path / "h2",
            health_client=health,
            parser=flaky,
            clock=clock,
        )
        replies = Replies()
        result = await hook.propose_from_text(origin(), "synthetic meal", replies)
        assert result is None
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_deadline_stops_second_attempt(self, tmp_path, clock, health):
        attempts = []

        async def slow_fail(text, image_path):
            attempts.append(1)
            clock.now += FOOD_PARSE_DEADLINE_SECONDS  # first attempt eats it all
            raise RuntimeError("slow")

        hook = SolFoodHook(
            state_dir=tmp_path / "s3",
            hermes_home=tmp_path / "h3",
            health_client=health,
            parser=slow_fail,
            clock=clock,
        )
        replies = Replies()
        result = await hook.propose_from_text(origin(), "synthetic meal", replies)
        assert result is None
        assert len(attempts) == 1


class TestMediaGate:
    @pytest.mark.asyncio
    async def test_album_denied_before_download(self, hook):
        replies = Replies()
        media = MediaDescriptor(
            kind="photo", file_size=100, width=10, height=10, media_group_id="grp"
        )
        decision = await hook.on_media_pre_download(SOL, origin(), media, replies)
        assert decision is HookDecision.DENY
        assert replies.messages == ["Please send one representative image."]

    @pytest.mark.asyncio
    async def test_second_photo_denied_while_proposal_active(self, hook):
        replies = Replies()
        proposal_id = await hook.propose_from_text(origin(), "synthetic meal", replies)
        assert proposal_id is not None
        media = MediaDescriptor(
            kind="photo", file_size=100, width=10, height=10, media_group_id=None
        )
        decision = await hook.on_media_pre_download(
            SOL, origin(update_id=1001), media, replies
        )
        assert decision is HookDecision.DENY

    @pytest.mark.asyncio
    async def test_clean_photo_continues(self, hook):
        replies = Replies()
        media = MediaDescriptor(
            kind="photo", file_size=100, width=10, height=10, media_group_id=None
        )
        decision = await hook.on_media_pre_download(SOL, origin(), media, replies)
        assert decision is HookDecision.CONTINUE

    @pytest.mark.asyncio
    async def test_non_photo_media_ignored(self, hook):
        replies = Replies()
        media = MediaDescriptor(
            kind="document", file_size=100, width=None, height=None, media_group_id=None
        )
        decision = await hook.on_media_pre_download(SOL, origin(), media, replies)
        assert decision is HookDecision.CONTINUE


class TestPhotoProposal:
    @pytest.mark.asyncio
    async def test_photo_cached_then_deleted(self, tmp_path, clock, health):
        import struct

        png = (
            b"\x89PNG\r\n\x1a\n"
            + struct.pack(">I", 13)
            + b"IHDR"
            + struct.pack(">II", 64, 64)
            + b"\x08\x02\x00\x00\x00"
        )
        seen_paths = []

        async def parser(text, image_path):
            seen_paths.append(image_path)
            assert image_path is not None and image_path.exists()
            mode = stat.S_IMODE(os.stat(image_path).st_mode)
            assert mode == 0o600
            return sample_candidates(1)

        hook = SolFoodHook(
            state_dir=tmp_path / "s4",
            hermes_home=tmp_path / "h4",
            health_client=health,
            parser=parser,
            clock=clock,
        )
        replies = Replies()
        proposal_id = await hook.propose_from_photo(origin(), png, None, replies)
        assert proposal_id is not None
        # Terminal parsing: the image is deleted immediately.
        assert seen_paths[0].exists() is False

    @pytest.mark.asyncio
    async def test_bad_bytes_rejected_and_never_cached(self, hook, tmp_path):
        replies = Replies()
        proposal_id = await hook.propose_from_photo(
            origin(), b"GIF89a" + b"\x00" * 64, None, replies
        )
        assert proposal_id is None
        food_dir = tmp_path / "state" / "food-images"
        assert list(food_dir.iterdir()) == []


class TestCallbacks:
    async def _confirm_flow(self, hook, health, *, cancel=False):
        replies = Replies()
        proposal_id = await hook.propose_from_text(origin(), "synthetic meal", replies)
        assert proposal_id is not None
        proposal = await hook._store.get(proposal_id)
        tokens = {
            record["action"]: token for token, record in proposal.tokens.items()
        }
        # Narrow to one candidate first (choice), which re-mints tokens.
        outcome_origin = origin(update_id=2000, message_id=None or 0)
        decision = await hook.on_callback(
            SOL, origin(update_id=2000), tokens["choice:0"], replies
        )
        assert decision is HookDecision.CONSUME
        proposal = await hook._store.get(proposal_id)
        tokens = {
            record["action"]: token
            for token, record in proposal.tokens.items()
            if not record["consumed"]
        }
        action = ACTION_CANCEL if cancel else ACTION_CONFIRM
        decision = await hook.on_callback(
            SOL, origin(update_id=2001), tokens[action], replies
        )
        assert decision is HookDecision.CONSUME
        return proposal_id, tokens, replies

    @pytest.mark.asyncio
    async def test_malformed_callback_denied(self, hook):
        replies = Replies()
        decision = await hook.on_callback(SOL, origin(), "sf1:short", replies)
        assert decision is HookDecision.DENY

    @pytest.mark.asyncio
    async def test_confirm_commits_exactly_once(self, hook, health):
        proposal_id, tokens, replies = await self._confirm_flow(hook, health)
        assert len(health.calls) == 1
        assert any(m.startswith("Logged") for m in replies.messages)
        # Second human tap on the consumed confirm: denied, no new commit.
        decision = await hook.on_callback(
            SOL, origin(update_id=2002), tokens[ACTION_CONFIRM], replies
        )
        assert decision is HookDecision.CONSUME
        assert len(health.calls) == 1

    @pytest.mark.asyncio
    async def test_duplicate_update_replays_without_second_commit(self, hook, health):
        proposal_id, tokens, replies = await self._confirm_flow(hook, health)
        # Exact same Telegram update re-delivered.
        decision = await hook.on_callback(
            SOL, origin(update_id=2001), tokens[ACTION_CONFIRM], replies
        )
        assert decision is HookDecision.CONSUME
        assert len(health.calls) == 1

    @pytest.mark.asyncio
    async def test_cancel_never_commits(self, hook, health):
        await self._confirm_flow(hook, health, cancel=True)
        assert health.calls == []

    @pytest.mark.asyncio
    async def test_confirm_requires_single_candidate(self, hook, health):
        replies = Replies()
        proposal_id = await hook.propose_from_text(origin(), "synthetic meal", replies)
        proposal = await hook._store.get(proposal_id)
        tokens = {r["action"]: t for t, r in proposal.tokens.items()}
        decision = await hook.on_callback(
            SOL, origin(update_id=2100), tokens[ACTION_CONFIRM], replies
        )
        assert decision is HookDecision.CONSUME
        assert health.calls == []
        assert "Pick one option first, then confirm." in replies.messages


class TestFrozenRetry:
    @pytest.mark.asyncio
    async def test_transport_loss_retries_identical_bytes(self, hook, health):
        health.error = HealthClientError("health_client_transport_error", retryable=True)
        replies = Replies()
        proposal_id = await hook.propose_from_text(origin(), "synthetic meal", replies)
        proposal = await hook._store.get(proposal_id)
        tokens = {r["action"]: t for t, r in proposal.tokens.items()}
        await hook.on_callback(SOL, origin(update_id=2200), tokens["choice:0"], replies)
        proposal = await hook._store.get(proposal_id)
        tokens = {
            r["action"]: t for t, r in proposal.tokens.items() if not r["consumed"]
        }
        await hook.on_callback(
            SOL, origin(update_id=2201), tokens[ACTION_CONFIRM], replies
        )
        assert len(health.calls) == 1
        first_bytes = health.calls[0]
        assert any("pending" in m for m in replies.messages)
        # Reconcile after "restart": identical frozen bytes, one effect.
        health.error = None
        health.replayed = True
        count = await hook.reconcile()
        assert count == 1
        assert len(health.calls) == 2
        assert health.calls[1] == first_bytes

    @pytest.mark.asyncio
    async def test_reconcile_survives_process_restart(self, tmp_path, clock, health):
        hook = SolFoodHook(
            state_dir=tmp_path / "s5",
            hermes_home=tmp_path / "h5",
            health_client=health,
            parser=default_parser,
            clock=clock,
        )
        health.error = HealthClientError("health_client_transport_error", retryable=True)
        replies = Replies()
        proposal_id = await hook.propose_from_text(origin(), "synthetic meal", replies)
        proposal = await hook._store.get(proposal_id)
        tokens = {r["action"]: t for t, r in proposal.tokens.items()}
        await hook.on_callback(SOL, origin(update_id=2300), tokens["choice:0"], replies)
        proposal = await hook._store.get(proposal_id)
        tokens = {
            r["action"]: t for t, r in proposal.tokens.items() if not r["consumed"]
        }
        await hook.on_callback(
            SOL, origin(update_id=2301), tokens[ACTION_CONFIRM], replies
        )
        first_bytes = health.calls[0]

        # New process (fresh hook over the same state dir).
        health.error = None
        reborn = SolFoodHook(
            state_dir=tmp_path / "s5",
            hermes_home=tmp_path / "h5",
            health_client=health,
            parser=default_parser,
            clock=clock,
        )
        count = await reborn.reconcile()
        assert count == 1
        assert health.calls[-1] == first_bytes
