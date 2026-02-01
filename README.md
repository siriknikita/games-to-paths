# games-to-paths

**Game → Symphony**: turn Gambler's Ruin random-walk trajectories into audio (WAV), MIDI, and symbolic output.

## What it does

- **Game**: Simulates [Gambler's Ruin](https://en.wikipedia.org/wiki/Gambler%27s_ruin): a random walk on states `0..K` with absorbing boundaries at 0 and K. You define the state space size (K), initial state (s0), and for each "run" a probability `p_up` of stepping +1 (else -1).
- **Mapping**: Each trajectory is turned into music:
  - **State → pitch**: position in `[0, K]` maps to frequency / MIDI note (center = base pitch, boundaries = higher/lower).
  - **Transition → duration**: step size influences note length.
  - **Boundary proximity → timbre**: closer to 0 or K ⇒ brighter tone; center ⇒ smoother.
- **Output**: For each run you get:
  - **WAV** – synthesized audio (mono, 16-bit).
  - **MIDI** – same mapping (notes, durations, velocities).
  - **Symbolic** – plain text trajectory (state sequence).
  - **Metrics** – trajectory length, state/transition entropy, compression ratio.

Same mapping, different rules (e.g. fair vs biased coin) produce different “symphonies.”

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for running and dependencies

## Setup

1. Clone or download the project.
2. Install dependencies with uv (from the project root):

   ```bash
   uv sync
   ```

3. **Config (recommended)**  
   A TOML config file is read **by default** from `config.toml` in the project directory. If it’s missing, built-in defaults are used.

   To use the example config:

   ```bash
   cp config.example.toml config.toml
   ```

   Then edit `config.toml` to set game parameters, runs, and mapping (see below).

## Usage

**Default (use `config.toml` if present):**

```bash
uv run python main.py
```

**Use a specific config file:**

```bash
uv run python main.py --config path/to/config.toml
```

**Ignore config and use built-in defaults:**

```bash
uv run python main.py --no-config
```

**Override individual options (these override config):**

```bash
uv run python main.py --K 50 --s0 25 --output-dir ./out
uv run python main.py --seed 42
```

### Options

| Option | Description |
|--------|-------------|
| `--config`, `-c` | Path to TOML config (default: `config.toml` in project directory) |
| `--no-config` | Don’t load any config file; use built-in defaults |
| `--K` | State space size (absorbing at 0 and K) |
| `--s0` | Initial state |
| `--seed` | Override random seed for all runs |
| `--output-dir`, `-o` | Output directory (default from config or `./output`) |

## Config file (TOML)

Example structure (see `config.example.toml`):

- **[game]**  
  `K` – state space 0..K; `s0` – initial state.

- **[[runs]]**  
  Each run has: `name`, `p_up` (probability of step +1), `seed`.

- **[mapping]**  
  Audio/MIDI: `sample_rate`, `base_freq`, `semitone_span`, `min_note_dur`, `max_note_dur`, `attack`, `release`, `volume`, `seed`.

- **output_dir** (optional)  
  Base directory for output; default is `./output`.

Output layout under that directory: `wav/`, `midi/`, `symbolic/`, with one WAV, one MIDI, and one symbolic file per run (e.g. `fair_trajectory.wav`, `fair_trajectory.mid`, `fair_trajectory_symbolic.txt`).

## Example

```bash
cp config.example.toml config.toml
uv run python main.py
```

Then open `output/wav/` (or your chosen `output_dir`) and listen to the generated WAVs; compare “fair” vs “biased_up” (or any runs you defined) to hear how different rules change the piece.
