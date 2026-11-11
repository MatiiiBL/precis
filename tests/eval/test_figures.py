from pathlib import Path

from precis.eval.figures import Panel, Series, small_multiples


def test_small_multiples_writes_a_png(tmp_path: Path) -> None:
    panels = [
        Panel("Papers per minute", "papers/min", [Series("throughput", [1, 2, 4])]),
        Panel(
            "Share of limit",
            "%",
            [Series("requests", [10, 20, 40]), Series("input tokens", [5, 15, 90])],
            reference=100,
            reference_label="limit",
        ),
        Panel("Errors", "%", [Series("429s", [0, 0, 0])]),
    ]
    path = small_multiples([4, 8, 16], panels, tmp_path / "img" / "bench.png", title="Bench")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
