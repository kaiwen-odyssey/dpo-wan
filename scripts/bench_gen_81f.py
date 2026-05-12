"""Quick wall-clock bench for one 5-second / 81-frame generation."""
from __future__ import annotations

import logging
import time

from dpo_wan import sampling
from dpo_wan.utils import set_seed, setup_logging

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    set_seed(0)
    log.info("loading Wan2.1-T2V-1.3B (T5 on CPU)...")
    t_load = time.time()
    wan = sampling.load_wan_t2v(t5_cpu=True)
    log.info("load wall: %.1fs", time.time() - t_load)

    spec = sampling.SampleSpec(
        size=(832, 480),
        frame_num=81,
        sampling_steps=15,
    )
    prompt = "A drone shot of a red Ferrari driving down a coastal mountain road"

    log.info("warm-up generation (excluded from timing)...")
    t0 = time.time()
    _ = sampling.generate_video(wan, prompt, seed=0, spec=spec)
    log.info("warm-up wall: %.1fs", time.time() - t0)

    times: list[float] = []
    for i in range(2):
        t0 = time.time()
        _ = sampling.generate_video(wan, prompt, seed=10 + i, spec=spec)
        dt = time.time() - t0
        times.append(dt)
        log.info("gen %d (81 f, 15 steps): %.1fs", i, dt)

    log.info("MEAN 81-frame gen wall: %.1fs (over %d runs)", sum(times) / len(times), len(times))


if __name__ == "__main__":
    main()
