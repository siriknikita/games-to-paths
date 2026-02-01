"""
Game -> Symphony testbed
- Game: Gambler's Ruin random walk with absorbing states 0 and K (guaranteed termination).
- Mapping: state -> pitch (frequency) / piano key (MIDI), transition magnitude -> duration, boundary proximity -> timbre brightness.
- Output: WAV audio, MIDI file, symbolic trajectory + basic structural metrics.

Run:
  uv run python main.py                    # use config.toml if present, else built-in defaults
  uv run python main.py --config other.toml
  uv run python main.py --K 50 --s0 25 -o ./out

Config: TOML file config.toml (in project dir) is read by default. Sections: [game], [[runs]], [mapping], output_dir.
CLI flags override config (--K, --s0, --seed, --output-dir).
"""

from __future__ import annotations

import argparse
import math
import random
import wave
import struct
import zlib
import tomllib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from midiutil import MIDIFile

# Output layout: output/{wav,midi,symbolic}/ (overridable via config)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def default_config() -> Dict[str, Any]:
    """Built-in defaults (no config file)."""
    return {
        "game": {"K": 40, "s0": 20},
        "runs": [
            {"name": "fair", "p_up": 0.50, "seed": 123},
            {"name": "biased_up", "p_up": 0.60, "seed": 123},
        ],
        "mapping": {
            "sample_rate": 44100,
            "base_freq": 220.0,
            "semitone_span": 28,
            "min_note_dur": 0.055,
            "max_note_dur": 0.16,
            "attack": 0.004,
            "release": 0.020,
            "volume": 0.23,
            "seed": 7,
        },
        "output_dir": str(DEFAULT_OUTPUT_DIR),
    }


def load_config(path: Path | None) -> Dict[str, Any]:
    """Load config from TOML; merge with defaults so partial configs work."""
    cfg = default_config()
    if path is not None and path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        if "game" in data:
            cfg["game"] = {**cfg["game"], **data["game"]}
        if "runs" in data:
            cfg["runs"] = data["runs"]
        if "mapping" in data:
            cfg["mapping"] = {**cfg["mapping"], **data["mapping"]}
        if "output_dir" in data:
            cfg["output_dir"] = data["output_dir"]
    return cfg


def parse_args() -> argparse.Namespace:
    default_config_path = PROJECT_ROOT / "config.toml"
    parser = argparse.ArgumentParser(
        description="Game -> Symphony: Gambler's Ruin trajectories to WAV/MIDI/symbolic."
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=default_config_path,
        help="Path to TOML config file (default: config.toml in project directory)",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Ignore config file and use built-in defaults only",
    )
    parser.add_argument("--K", type=int, default=None, help="State space size (absorbing at 0 and K)")
    parser.add_argument("--s0", type=int, default=None, help="Initial state")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for all runs (optional)")
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="Output directory (output/wav, etc.)")
    args = parser.parse_args()
    if args.no_config:
        args.config = None
    return args


# -----------------------------
# 1) Game model: Gambler's Ruin
# -----------------------------

@dataclass(frozen=True)
class GamblerRuinGame:
    K: int                 # absorbing boundaries at 0 and K
    p_up: float = 0.5      # probability to step +1 (else -1)

    def simulate(self, s0: int, max_steps: int = 1_000_000, seed: int | None = None) -> List[int]:
        """
        Returns state trajectory [s0, s1, ..., sN] with sN in {0,K}.
        Guaranteed termination almost surely for 0<p<1 in finite state space;
        we keep a max_steps guard anyway.
        """
        if not (0 <= s0 <= self.K):
            raise ValueError("s0 must be within [0, K].")
        if not (0.0 < self.p_up < 1.0):
            raise ValueError("p_up must be in (0,1).")

        rng = random.Random(seed)
        s = s0
        traj = [s]

        for _ in range(max_steps):
            if s == 0 or s == self.K:
                return traj
            step = 1 if (rng.random() < self.p_up) else -1
            s = s + step
            traj.append(s)

        # If we somehow hit max_steps (should be extremely rare for moderate K), return what we have.
        return traj


# -----------------------------------------
# 2) Musical mapping: state -> audio events
# -----------------------------------------

@dataclass(frozen=True)
class MusicMapping:
    sample_rate: int = 44100
    base_freq: float = 220.0         # A3
    semitone_span: int = 24          # map state range into 2 octaves by default
    min_note_dur: float = 0.06       # seconds
    max_note_dur: float = 0.18       # seconds
    attack: float = 0.006            # seconds
    release: float = 0.020           # seconds
    volume: float = 0.25             # master volume (0..1)
    seed: int = 7                    # for slight deterministic variation in partials
    # Piano / MIDI: same pitch mapping as state_to_freq (A3 = 57)
    base_midi_note: int = 57         # A3
    midi_note_min: int = 21         # A0
    midi_note_max: int = 108        # C8

    def state_to_freq(self, s: int, K: int) -> float:
        """
        Map integer state s in [0,K] to a frequency using equal temperament steps.
        """
        if K <= 0:
            return self.base_freq
        x = s / K  # 0..1
        # Map to [-span/2, +span/2] semitones around base
        semis = (x - 0.5) * self.semitone_span
        return self.base_freq * (2.0 ** (semis / 12.0))

    def transition_to_duration(self, s_prev: int, s_next: int) -> float:
        """
        Transition magnitude -> duration (bigger jump => longer note).
        For gambler's ruin steps are +/-1, but you can plug other games in later.
        """
        jump = abs(s_next - s_prev)
        # Normalize jump: for this game it's 1, but keep general
        jump_norm = min(1.0, jump / 5.0)
        return self.min_note_dur + (self.max_note_dur - self.min_note_dur) * jump_norm

    def brightness(self, s: int, K: int) -> float:
        """
        Boundary proximity -> brightness (0..1).
        Near boundaries => brighter/edgier; center => smoother.
        """
        if K <= 0:
            return 0.5
        x = s / K  # 0..1
        dist_to_center = abs(x - 0.5) * 2.0  # 0 at center, 1 at boundaries
        return max(0.0, min(1.0, dist_to_center))

    def state_to_midi_note(self, s: int, K: int) -> int:
        """
        Map integer state s in [0,K] to a MIDI note (piano key).
        Uses same formula as state_to_freq: semis = (s/K - 0.5) * semitone_span.
        Clamped to piano range [midi_note_min, midi_note_max].
        """
        if K <= 0:
            return self.base_midi_note
        x = s / K  # 0..1
        semis = (x - 0.5) * self.semitone_span
        midi = self.base_midi_note + round(semis)
        return max(self.midi_note_min, min(self.midi_note_max, midi))


# -----------------------------
# 3) WAV synthesis (pure python)
# -----------------------------

def adsr_envelope(t: float, dur: float, attack: float, release: float) -> float:
    """
    Simple attack-release envelope (no sustain/decay) for clarity.
    """
    if dur <= 1e-9:
        return 0.0
    a = min(attack, dur)
    r = min(release, dur - a) if dur > a else 0.0

    if t < 0.0 or t > dur:
        return 0.0
    if t <= a:
        return t / a if a > 1e-9 else 1.0
    if t >= dur - r:
        return (dur - t) / r if r > 1e-9 else 0.0
    return 1.0


def synth_note(freq: float, dur: float, sr: int, volume: float, attack: float, release: float,
               brightness: float, rng: random.Random) -> List[float]:
    """
    Synthesize a note as a sum of partials; brightness controls high harmonics weight.
    """
    n = int(max(1, round(dur * sr)))
    out = [0.0] * n

    # Partials: 1..6
    # Brightness increases weights of higher partials.
    partials = 6
    for i in range(n):
        t = i / sr
        env = adsr_envelope(t, dur, attack, release)
        # fundamental + harmonics
        s = 0.0
        for k in range(1, partials + 1):
            # Base amplitude decays with k, but brightness boosts higher k
            base_amp = 1.0 / (k ** 1.2)
            boost = (0.25 + 0.75 * brightness) ** (k - 1)  # more brightness => less decay
            # Tiny deterministic detune to avoid sterile tone
            detune = 1.0 + (rng.random() - 0.5) * 0.0006
            s += base_amp * boost * math.sin(2.0 * math.pi * (freq * k * detune) * t)
        out[i] = volume * env * s

    # soft clip normalization for this note
    peak = max(1e-9, max(abs(x) for x in out))
    scale = 0.98 / peak
    return [x * scale for x in out]


def render_trajectory_to_wav(traj: List[int], K: int, mapping: MusicMapping,
                             wav_path: str, symbolic_path: str) -> None:
    rng = random.Random(mapping.seed)

    # Build audio by concatenating notes for each transition
    audio: List[float] = []

    # Save symbolic too
    with open(symbolic_path, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, traj)) + "\n")

    for i in range(1, len(traj)):
        s_prev, s_next = traj[i - 1], traj[i]
        freq = mapping.state_to_freq(s_next, K)
        dur = mapping.transition_to_duration(s_prev, s_next)
        bright = mapping.brightness(s_next, K)

        note = synth_note(
            freq=freq,
            dur=dur,
            sr=mapping.sample_rate,
            volume=mapping.volume,
            attack=mapping.attack,
            release=mapping.release,
            brightness=bright,
            rng=rng
        )
        audio.extend(note)

    # Convert to 16-bit PCM
    pcm = bytearray()
    for x in audio:
        # final soft clip
        x = max(-1.0, min(1.0, x))
        pcm += struct.pack("<h", int(x * 32767))

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(mapping.sample_rate)
        wf.writeframes(pcm)


def render_trajectory_to_midi(
    traj: List[int],
    K: int,
    mapping: MusicMapping,
    midi_path: str,
    *,
    ticks_per_beat: int = 480,
    tempo_bpm: int = 120,
) -> None:
    """
    Render trajectory to a MIDI file. Same mapping as WAV: state -> MIDI note,
    transition -> duration. Velocity derived from brightness (boundary proximity).
    """
    mf = MIDIFile(1, ticks_per_quarternote=ticks_per_beat)
    mf.addTempo(0, 0, tempo_bpm)

    # Time in beats (MIDIUtil uses beats, not ticks)
    time_beats = 0.0
    for i in range(1, len(traj)):
        s_prev, s_next = traj[i - 1], traj[i]
        midi_note = mapping.state_to_midi_note(s_next, K)
        duration_sec = mapping.transition_to_duration(s_prev, s_next)
        duration_beats = duration_sec * tempo_bpm / 60.0
        bright = mapping.brightness(s_next, K)
        velocity = int(40 + 80 * bright)  # 40..120

        mf.addNote(
            track=0,
            channel=0,
            pitch=midi_note,
            time=time_beats,
            duration=duration_beats,
            volume=velocity,
        )
        time_beats += duration_beats

    with open(midi_path, "wb") as f:
        mf.writeFile(f)


# -----------------------------
# 4) Structural metrics
# -----------------------------

def shannon_entropy(probabilities: List[float]) -> float:
    e = 0.0
    for p in probabilities:
        if p > 0.0:
            e -= p * math.log(p, 2)
    return e


def trajectory_metrics(traj: List[int]) -> Dict[str, float]:
    n = len(traj)
    if n <= 1:
        return {
            "length": float(n),
            "state_entropy_bits": 0.0,
            "transition_entropy_bits": 0.0,
            "compression_ratio": 1.0,
        }

    # State distribution
    counts: Dict[int, int] = {}
    for s in traj:
        counts[s] = counts.get(s, 0) + 1
    probs = [c / n for c in counts.values()]
    state_H = shannon_entropy(probs)

    # Transition distribution
    tcounts: Dict[Tuple[int, int], int] = {}
    for i in range(1, n):
        a = (traj[i - 1], traj[i])
        tcounts[a] = tcounts.get(a, 0) + 1
    tprobs = [c / (n - 1) for c in tcounts.values()]
    trans_H = shannon_entropy(tprobs)

    # Compressibility proxy on symbolic sequence
    sym = (" ".join(map(str, traj))).encode("utf-8")
    comp = zlib.compress(sym, level=9)
    compression_ratio = len(comp) / max(1, len(sym))

    return {
        "length": float(n),
        "state_entropy_bits": state_H,
        "transition_entropy_bits": trans_H,
        "compression_ratio": compression_ratio,
    }


# -----------------------------
# 5) Main experiment
# -----------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.K is not None:
        cfg["game"]["K"] = args.K
    if args.s0 is not None:
        cfg["game"]["s0"] = args.s0
    if args.output_dir is not None:
        cfg["output_dir"] = str(args.output_dir)
    if args.seed is not None:
        for run in cfg["runs"]:
            run["seed"] = args.seed

    K = cfg["game"]["K"]
    s0 = cfg["game"]["s0"]
    output_dir = Path(cfg["output_dir"])
    wav_dir = output_dir / "wav"
    midi_dir = output_dir / "midi"
    symbolic_dir = output_dir / "symbolic"

    games = [
        (r["name"], GamblerRuinGame(K=K, p_up=float(r["p_up"])), int(r["seed"]))
        for r in cfg["runs"]
    ]
    mapping = MusicMapping(**cfg["mapping"])

    # Ensure output dirs exist
    wav_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)
    symbolic_dir.mkdir(parents=True, exist_ok=True)

    for name, game, seed in games:
        traj = game.simulate(s0=s0, seed=seed)
        wav_path = str(wav_dir / f"{name}_trajectory.wav")
        symbolic_path = str(symbolic_dir / f"{name}_trajectory_symbolic.txt")
        midi_path = str(midi_dir / f"{name}_trajectory.mid")

        render_trajectory_to_wav(traj, K=K, mapping=mapping, wav_path=wav_path, symbolic_path=symbolic_path)
        render_trajectory_to_midi(traj, K=K, mapping=mapping, midi_path=midi_path)

        m = trajectory_metrics(traj)
        print(f"\n=== {name} ===")
        print(f"trajectory length: {int(m['length'])} states")
        print(f"state entropy: {m['state_entropy_bits']:.4f} bits")
        print(f"transition entropy: {m['transition_entropy_bits']:.4f} bits")
        print(f"compression ratio (lower => more structured): {m['compression_ratio']:.4f}")
        print(f"wrote: {wav_path}")
        print(f"wrote: {symbolic_path}")
        print(f"wrote: {midi_path}")

    print("\nDone. Listen to the WAVs and compare: same mapping, different rules => different 'symphonies'.")


if __name__ == "__main__":
    main()
