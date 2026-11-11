# Throughput: 2,000 papers, offline

Simulated limits: 4,000 requests, 2,000,000 input tokens and 400,000 output tokens per minute. Latency: 2.0 s to first token plus 0.02 s per output token, log-normal jitter 0.35. 'Final window' is the adaptive concurrency limit at the end of the run.

| Ceiling | Papers/min | Requests/min | RPM used | ITPM used | OTPM used | 429s | Final window | p50 paper (min) | p95 paper (min) | Run (sim. min) | Wall (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 7.9 | 40 | 1% | 5% | 2% | 0.00% | 4 | 0.9 | 1.7 | 252.4 | 28 |
| 8 | 15.9 | 80 | 2% | 9% | 4% | 0.00% | 8 | 0.9 | 1.7 | 126.1 | 29 |
| 16 | 31.6 | 159 | 4% | 18% | 8% | 0.00% | 16 | 0.9 | 1.6 | 63.4 | 29 |
| 32 | 62.9 | 316 | 8% | 37% | 16% | 0.00% | 32 | 0.9 | 1.7 | 31.8 | 29 |
| 64 | 123.1 | 618 | 15% | 72% | 31% | 0.00% | 64 | 0.9 | 1.7 | 16.2 | 29 |
| 128 | 165.1 | 829 | 21% | 96% | 42% | 0.00% | 128 | 1.3 | 2.4 | 12.1 | 30 |
| 256 | 166.3 | 835 | 21% | 97% | 42% | 0.00% | 256 | 2.4 | 4.7 | 12.0 | 30 |
