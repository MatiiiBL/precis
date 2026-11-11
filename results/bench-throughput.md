# Throughput: 2,000 papers, offline

Simulated limits: 4,000 requests, 2,000,000 input tokens and 400,000 output tokens per minute. Latency: 2.0 s to first token plus 0.02 s per output token, log-normal jitter 0.35. 'Final window' is the adaptive concurrency limit at the end of the run.

| Ceiling | Papers/min | Requests/min | RPM used | ITPM used | OTPM used | 429s | Final window | p50 paper (min) | p95 paper (min) | Run (sim. min) | Wall (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 7.9 | 40 | 1% | 5% | 2% | 0.00% | 4 | 0.9 | 1.7 | 252.4 | 28 |
| 8 | 15.9 | 80 | 2% | 9% | 4% | 0.00% | 8 | 0.9 | 1.7 | 126.1 | 29 |
| 16 | 31.7 | 159 | 4% | 19% | 8% | 0.00% | 16 | 0.9 | 1.6 | 63.1 | 28 |
| 32 | 61.1 | 307 | 8% | 36% | 15% | 0.00% | 32 | 0.9 | 1.7 | 32.8 | 28 |
| 64 | 108.1 | 543 | 14% | 63% | 27% | 0.00% | 64 | 0.9 | 2.2 | 18.5 | 29 |
| 128 | 130.6 | 656 | 16% | 76% | 33% | 0.00% | 128 | 1.5 | 3.7 | 15.3 | 30 |
| 256 | 132.0 | 663 | 17% | 77% | 33% | 0.00% | 142 | 2.9 | 6.1 | 15.1 | 32 |
