"""Tests for the Squeezelite player."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from aioslimproto.models import EventType as SlimEventType
from aioslimproto.models import PlayerState as SlimPlayerState
from aioslimproto.models import SlimEvent
from music_assistant_models.enums import MediaType, PlaybackState, PlayerType
from music_assistant_models.player import PlayerMedia

from music_assistant.constants import (
    CONF_ENTRY_PREFER_WAV_FOR_LIVE_SOURCES_DEFAULT_ENABLED,
    CONF_OUTPUT_CODEC,
    CONF_PREFER_WAV_FOR_LIVE_SOURCES,
)
from music_assistant.providers.squeezelite.player import SqueezelitePlayer, is_protocol_only_device
from tests.common import use_real_create_task


async def test_squeezelite_prefers_wav_for_live_sources_by_default() -> None:
    """Squeezelite players default to the known-compatible low-latency WAV path."""

    async def _no_library_items(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
        return
        yield  # pragma: no cover - makes this an async generator

    player = SqueezelitePlayer.__new__(SqueezelitePlayer)
    player.mass = mass = MagicMock()
    mass.music.playlists.iter_library_items = _no_library_items
    mass.music.radio.iter_library_items = _no_library_items

    entries = await player.get_config_entries()
    entry = next(entry for entry in entries if entry.key == CONF_PREFER_WAV_FOR_LIVE_SOURCES)

    assert (
        entry.default_value == CONF_ENTRY_PREFER_WAV_FOR_LIVE_SOURCES_DEFAULT_ENABLED.default_value
    )


def _member_player(*, prefer_wav_for_live_sources: bool) -> MagicMock:
    """Return a mocked member player with the given effective WAV preference."""
    member = MagicMock()
    member.config.get_value.side_effect = lambda key, default=None: (
        prefer_wav_for_live_sources if key == CONF_PREFER_WAV_FOR_LIVE_SOURCES else default
    )
    return member


def _player_with_mocked_mass() -> tuple[SqueezelitePlayer, MagicMock]:
    """Return an uninitialized SqueezelitePlayer alongside its mocked mass instance."""
    player = SqueezelitePlayer.__new__(SqueezelitePlayer)
    player.mass = mass = MagicMock()
    return player, mass


def test_group_audio_source_uses_wav_when_member_prefers_it() -> None:
    """A sync group member that opted into low-latency WAV gets WAV for AudioSource media."""
    player, mass = _player_with_mocked_mass()
    mass.players.get_player.return_value = _member_player(prefer_wav_for_live_sources=True)

    codec = player._get_member_output_codec(
        "member_1", PlayerMedia(uri="fake://x", media_type=MediaType.AUDIO_SOURCE)
    )

    assert codec == "wav"
    mass.config.get_raw_player_config_value.assert_not_called()


def test_group_audio_source_falls_back_to_output_codec_when_disabled() -> None:
    """A sync group member that disabled the WAV preference uses its configured output codec."""
    player, mass = _player_with_mocked_mass()
    mass.players.get_player.return_value = _member_player(prefer_wav_for_live_sources=False)
    mass.config.get_raw_player_config_value.return_value = "aac"

    codec = player._get_member_output_codec(
        "member_1", PlayerMedia(uri="fake://x", media_type=MediaType.AUDIO_SOURCE)
    )

    assert codec == "aac"
    mass.config.get_raw_player_config_value.assert_called_once_with(
        "member_1", CONF_OUTPUT_CODEC, "flac"
    )


def test_group_regular_track_ignores_wav_preference() -> None:
    """Regular track playback in a sync group always uses the member's configured codec."""
    player, mass = _player_with_mocked_mass()
    mass.config.get_raw_player_config_value.return_value = "flac"

    codec = player._get_member_output_codec(
        "member_1", PlayerMedia(uri="fake://x", media_type=MediaType.TRACK)
    )

    assert codec == "flac"
    mass.players.get_player.assert_not_called()


@pytest.mark.parametrize(
    ("device_model", "expected"),
    [
        # generic squeezelite (software) players are full players
        ("SqueezeLite", False),
        ("SqueezeLite-HA-Addon", False),
        ("SqueezePlay", False),
        ("SqueezeESP32", False),
        # hardware players are full players
        ("Squeezebox Boom", False),
        ("Transporter", False),
        # unknown/absent model info defaults to a full player
        ("", False),
        ("Unknown", False),
        # WiiM/LinkPlay devices use squeezelite as a secondary protocol
        ("WiiM Player", True),
        ("wiim mini", True),
        # LMS bridge tools represent devices that are already players themselves
        ("RaopBridge", True),
        ("CastBridge", True),
        ("UPnPBridge", True),
    ],
)
def test_is_protocol_only_device(device_model: str, expected: bool) -> None:
    """Test protocol-only device detection based on the reported device model."""
    assert is_protocol_only_device(device_model) is expected


async def test_regroup_supersedes_pending_resume() -> None:
    """A member removed and re-added while the leader is busy restarts the leader once."""
    player, mass = _player_with_mocked_mass()
    player._player_id = "leader"
    player._attr_group_members = ["leader", "member"]
    player._attr_playback_state = PlaybackState.PLAYING
    use_real_create_task(mass)
    member = MagicMock()
    member.player_id = "member"
    member.state = SlimPlayerState.STOPPED
    mass.players.get_player.return_value = member
    # the playback lock an announcement holds on the leader while it plays
    leader_busy = asyncio.Event()
    resumed: list[str] = []

    async def _cmd_resume(player_id: str) -> None:
        await leader_busy.wait()
        resumed.append(player_id)

    mass.players.cmd_resume = _cmd_resume

    with (
        patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None),
        patch.object(SqueezelitePlayer, "state", new_callable=PropertyMock),
        patch.object(player, "_get_sync_clients", return_value=[member]),
        patch.object(player, "update_state"),
    ):
        await player.set_members(player_ids_to_remove=["member"])
        await player.set_members(player_ids_to_add=["member"])
        leader_busy.set()
        await asyncio.gather(*mass._tracked_tasks.values(), return_exceptions=True)

    assert resumed == ["leader"]


async def test_play_media_resets_elapsed_anchor() -> None:
    """A new stream does not carry the previous stream's position."""
    player, mass = _player_with_mocked_mass()
    player.multi_client_stream = None
    player.client = MagicMock(elapsed_seconds=0)
    player._provider = MagicMock(mass=mass)
    player._attr_elapsed_time = 3.9
    player._attr_elapsed_time_last_updated = time.time() - 4
    mass.streams.resolve_stream_url = AsyncMock(return_value="http://127.0.0.1:8097/x.flac")

    with (
        patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None),
        patch.object(
            SqueezelitePlayer, "group_members", new_callable=PropertyMock, return_value=[]
        ),
        patch.object(SqueezelitePlayer, "player_id", new_callable=PropertyMock, return_value="p1"),
        patch.object(player, "_handle_play_url_for_slimplayer", new=AsyncMock()),
        patch.object(player, "update_attributes"),
        patch.object(player, "update_state"),
    ):
        await player.play_media(
            PlayerMedia(uri="fake://x", media_type=MediaType.TRACK, source_id="queue_1")
        )

    assert player._attr_elapsed_time == 0
    assert time.time() - player._attr_elapsed_time_last_updated < 1


async def test_play_media_drops_heartbeat_from_before_the_flush() -> None:
    """An old-stream heartbeat that lands before the flush does not survive the restart."""
    player, mass = _player_with_mocked_mass()
    player.multi_client_stream = None
    player._extra_data = {}
    player.client = client = MagicMock(elapsed_seconds=0)
    client.play_url = AsyncMock()
    player._provider = MagicMock(mass=mass)
    mass.player_queues.get.return_value = None

    async def _resolve_with_heartbeat(*_args: object) -> str:
        player._attr_elapsed_time = 3.9
        return "http://127.0.0.1:8097/x.flac"

    mass.streams.resolve_stream_url = _resolve_with_heartbeat

    with (
        patch.object(SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=None),
        patch.object(
            SqueezelitePlayer, "group_members", new_callable=PropertyMock, return_value=[]
        ),
        patch.object(SqueezelitePlayer, "player_id", new_callable=PropertyMock, return_value="p1"),
    ):
        await player.play_media(
            PlayerMedia(uri="fake://x", media_type=MediaType.TRACK, source_id="queue_1")
        )

    assert player._attr_elapsed_time == 0


def _stalled_player() -> tuple[SqueezelitePlayer, MagicMock]:
    """Return a single player whose client just reported an output underrun."""
    player, mass = _player_with_mocked_mass()
    use_real_create_task(mass)
    player.logger = MagicMock()
    player.multi_client_stream = None
    player._extra_data = {}
    player._stream_generation = 0
    player._provider = MagicMock(mass=mass)
    player.client = client = MagicMock(state=SlimPlayerState.BUFFERING, jiffies=123456)
    client.connected = True
    client.unpause_at = AsyncMock()
    client.play_url = AsyncMock()
    client.stop = AsyncMock()
    mass.players.get_player.return_value = player
    mass.player_queues.get.return_value = None
    mass.streams.resolve_stream_url = AsyncMock(return_value="http://127.0.0.1:8097/x.flac")
    return player, mass


async def _underrun(
    player: SqueezelitePlayer,
    mass: MagicMock,
    during_wait: Any = None,
    group_members: list[str] | None = None,
    synced_to: str | None = None,
) -> None:
    """Report an output underrun and wait for the provider to handle it."""
    with (
        patch("music_assistant.providers.squeezelite.player.UNDERRUN_RESUME_DELAY", 0.05),
        patch("music_assistant.providers.squeezelite.player.UNDERRUN_RESUME_CHECK_DELAY", 0.05),
        patch.object(SqueezelitePlayer, "display_name", new_callable=PropertyMock),
        patch.object(
            SqueezelitePlayer, "synced_to", new_callable=PropertyMock, return_value=synced_to
        ),
        patch.object(
            SqueezelitePlayer,
            "group_members",
            new_callable=PropertyMock,
            return_value=group_members or [],
        ),
        patch.object(SqueezelitePlayer, "player_id", new_callable=PropertyMock, return_value="p1"),
        patch.object(player, "update_attributes"),
        patch.object(player, "update_state"),
    ):
        player.handle_slim_event(SlimEvent(SlimEventType.PLAYER_OUTPUT_UNDERRUN, "p1"))
        if during_wait is not None:
            await during_wait()
        results = await asyncio.gather(*mass._tracked_tasks.values(), return_exceptions=True)
    # a replaced task ends cancelled, which is not an Exception; anything else is a bug
    assert not [result for result in results if isinstance(result, Exception)]


async def test_output_underrun_resumes_playback() -> None:
    """A player that ran out of audio is told to resume once it has rebuffered."""
    player, mass = _stalled_player()

    await _underrun(player, mass)

    player.client.unpause_at.assert_awaited_once_with(123456)


def _logged(log_method: MagicMock) -> list[str]:
    """Return the format strings passed to a mocked logger method."""
    return [call.args[0] for call in log_method.call_args_list]


async def test_output_underrun_logs_a_confirmed_resume() -> None:
    """A resume the player confirms is logged as recovered, not as a failure."""
    player, mass = _stalled_player()

    async def _confirm_resume(_timestamp: int) -> None:
        player.client.state = SlimPlayerState.PLAYING

    player.client.unpause_at.side_effect = _confirm_resume

    await _underrun(player, mass)

    assert any("resumed playback" in message for message in _logged(player.logger.info))
    assert not any("did not resume" in message for message in _logged(player.logger.warning))


async def test_output_underrun_logs_a_resume_the_player_ignores() -> None:
    """A player that stays silent after the resume is reported, so a failed recovery shows."""
    player, mass = _stalled_player()

    await _underrun(player, mass)

    assert any("did not resume" in message for message in _logged(player.logger.warning))


@pytest.mark.parametrize("new_state", [SlimPlayerState.PAUSED, SlimPlayerState.STOPPED])
async def test_output_underrun_does_not_resume_after_pause_or_stop(
    new_state: SlimPlayerState,
) -> None:
    """A pause or stop while the player rebuffers is not undone."""
    player, mass = _stalled_player()

    async def _leave_buffering() -> None:
        player.client.state = new_state

    await _underrun(player, mass, during_wait=_leave_buffering)

    player.client.unpause_at.assert_not_awaited()


async def test_output_underrun_does_not_resume_a_new_stream() -> None:
    """A stream started while the player rebuffers is left to its own buffer-ready start."""
    player, mass = _stalled_player()

    async def _start_new_stream() -> None:
        await player.play_media(
            PlayerMedia(uri="fake://x", media_type=MediaType.TRACK, source_id="queue_1")
        )

    await _underrun(player, mass, during_wait=_start_new_stream)

    player.client.unpause_at.assert_not_awaited()


async def test_output_underrun_does_not_resume_after_stop_with_late_underrun() -> None:
    """An underrun report that crosses a stop does not restart the stopped player."""
    player, mass = _stalled_player()
    player._attr_type = PlayerType.PLAYER

    async def _stop() -> None:
        # the late underrun report left the client in BUFFERING despite the stop
        await player.stop()

    await _underrun(player, mass, during_wait=_stop)

    player.client.stop.assert_awaited_once()
    player.client.unpause_at.assert_not_awaited()


async def test_output_underrun_still_resumes_after_enqueue() -> None:
    """Enqueueing the next track does not replace the stalled stream, so it is still resumed."""
    player, mass = _stalled_player()

    async def _enqueue() -> None:
        await player.enqueue_next_media(
            PlayerMedia(uri="fake://next", media_type=MediaType.TRACK, source_id="queue_1")
        )

    await _underrun(player, mass, during_wait=_enqueue)

    player.client.unpause_at.assert_awaited_once_with(123456)


async def test_output_underrun_does_not_resume_a_disconnected_player() -> None:
    """A player that disconnected while rebuffering is left alone."""
    player, mass = _stalled_player()

    async def _disconnect() -> None:
        player.client.connected = False

    await _underrun(player, mass, during_wait=_disconnect)

    player.client.unpause_at.assert_not_awaited()
    assert not any("did not resume" in message for message in _logged(player.logger.warning))


async def test_newer_output_underrun_replaces_a_pending_resume() -> None:
    """A second underrun while a resume is pending results in a single resume."""
    player, mass = _stalled_player()

    async def _underrun_again() -> None:
        player.handle_slim_event(SlimEvent(SlimEventType.PLAYER_OUTPUT_UNDERRUN, "p1"))

    await _underrun(player, mass, during_wait=_underrun_again)

    player.client.unpause_at.assert_awaited_once_with(123456)


async def test_resume_is_not_credited_to_a_new_stream() -> None:
    """A new stream that starts after the resume was sent is not reported as recovered."""
    player, mass = _stalled_player()

    async def _ignore_resume_and_start_new_stream(_timestamp: int) -> None:
        await player.play_media(
            PlayerMedia(uri="fake://x", media_type=MediaType.TRACK, source_id="queue_1")
        )
        player.client.state = SlimPlayerState.PLAYING

    player.client.unpause_at.side_effect = _ignore_resume_and_start_new_stream

    await _underrun(player, mass)

    assert not any("resumed playback" in message for message in _logged(player.logger.info))
    assert not any("did not resume" in message for message in _logged(player.logger.warning))


@pytest.mark.parametrize(
    ("group_members", "synced_to"), [(["p1", "p2"], None), ([], "leader")], ids=["leader", "member"]
)
async def test_output_underrun_in_sync_group_is_not_resumed(
    group_members: list[str], synced_to: str | None
) -> None:
    """A group player is not resumed on its own, which would break sync, but it is reported."""
    player, mass = _stalled_player()

    await _underrun(player, mass, group_members=group_members, synced_to=synced_to)

    player.client.unpause_at.assert_not_awaited()
    assert any("sync group" in message for message in _logged(player.logger.warning))
