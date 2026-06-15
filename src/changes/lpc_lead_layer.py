"""LPC Lead Layer note computation for Tracks 9–16 (v0.3.0 beta).

Each chord cell's Local Pitch Collection (selected scale pitch classes) is
placed onto Tracks 9–16 in ascending pitch order starting from a user-chosen
one-octave range.  Exactly 8 MIDI notes are returned: if fewer than 8 pitch
class instances exist in the range the sequence continues into the next octave;
if more than 8 exist the lowest 8 are taken.
"""

from __future__ import annotations

LPC_LEAD_SLOT_COUNT = 8
LPC_LEAD_FIRST_TRACK = 9
LPC_LEAD_LAST_TRACK = 16

_VOICE_ID_PREFIX = "lpc_lead_slot_"


def lpc_lead_voice_ids() -> tuple[str, ...]:
    """Return the canonical voice IDs for LPC Lead Layer slots 1–8."""
    return tuple(f"{_VOICE_ID_PREFIX}{i}" for i in range(1, LPC_LEAD_SLOT_COUNT + 1))


def lpc_lead_track_for_slot(slot_1based: int) -> int:
    """Return the Digitone track number (9–16) for a 1-based slot index."""
    if not 1 <= slot_1based <= LPC_LEAD_SLOT_COUNT:
        raise ValueError(f"slot must be 1..{LPC_LEAD_SLOT_COUNT}, got {slot_1based}")
    return LPC_LEAD_FIRST_TRACK + slot_1based - 1


def lpc_lead_range_start_midi(range_root_pc: int, octave: int) -> int:
    """Convert (pitch class, octave) to a MIDI note number.

    Uses the standard convention where C4 = 60 (MIDI = (octave+1)*12 + pc).
    """
    if not 0 <= range_root_pc <= 11:
        raise ValueError(f"range_root_pc must be 0..11, got {range_root_pc}")
    return (octave + 1) * 12 + range_root_pc


def compute_lpc_lead_notes(
    pitch_classes: frozenset[int],
    range_start_midi: int,
) -> tuple[int, ...]:
    """Compute exactly 8 MIDI notes for LPC Lead Layer (Tracks 9–16).

    Starting from range_start_midi, walks upward and collects MIDI notes
    whose pitch class is in pitch_classes until 8 notes are gathered.
    This naturally:
    - Places all matching notes in the first octave in ascending order
    - Continues into the octave above when fewer than 8 are found there
    - Stops at 8 when more than 8 pitch class instances are available
    """
    if not pitch_classes:
        raise ValueError("pitch_classes must not be empty")
    notes: list[int] = []
    midi = range_start_midi
    # Walk up until we have 8 notes; limit scan to avoid infinite loop on empty set
    scan_limit = range_start_midi + 128
    while len(notes) < LPC_LEAD_SLOT_COUNT and midi < scan_limit:
        if midi % 12 in pitch_classes:
            notes.append(midi)
        midi += 1
    if len(notes) < LPC_LEAD_SLOT_COUNT:
        raise ValueError(
            f"Could not find {LPC_LEAD_SLOT_COUNT} notes from pitch_classes={pitch_classes!r} "
            f"within 128 semitones of range_start_midi={range_start_midi}"
        )
    return tuple(notes)
