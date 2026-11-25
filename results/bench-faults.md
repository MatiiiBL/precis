# Faults and crash recovery: passed

Injected per request: 429 4%, 529 2%, 500 2%, timeout 1%. The first run was killed with SIGKILL; a second run resumed from the store it left behind.

|  | Value |
| --- | --- |
| Papers | 500 |
| Papers finished when the first run was killed | 202 |
| Stages running at the kill | 64 |
| Requests in flight at the kill | 3 |
| Requests answered in the killed run | 1,092 |
| Requests in the resumed run | 1,427 |
| of which served from the cache | 48 |
| of which in flight at the kill, sent again | 3 |
| Work kept: stages finished before the kill | $17.19 |
| Work kept: replayed from the cache on resume | $0.30 |
| Total cost of both runs | $40.71 |
| Retries in both runs | 257 |
| Faults injected in the resumed run | 66 x 429, 27 x 529, 38 x 500, 17 x timeout |
| Papers done / failed / lost | 500 / 0 / 0 |
| Requests paid for twice | 0 |
