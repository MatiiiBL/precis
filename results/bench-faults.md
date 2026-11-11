# Faults and crash recovery: passed

Injected per request: 429 4%, 529 2%, 500 2%, timeout 1%. The first run was killed with SIGKILL; a second run resumed from the store it left behind.

|  | Value |
| --- | --- |
| Papers | 500 |
| Papers finished when the first run was killed | 201 |
| Stages running at the kill | 64 |
| Requests in flight at the kill | 1 |
| Requests answered in the killed run | 1,091 |
| Requests in the resumed run | 1,426 |
| of which served from the cache | 46 |
| of which in flight at the kill, sent again | 1 |
| Work kept: stages finished before the kill | $16.92 |
| Work kept: replayed from the cache on resume | $0.26 |
| Total cost of both runs | $40.03 |
| Retries in both runs | 256 |
| Faults injected in the resumed run | 66 x 429, 27 x 529, 38 x 500, 17 x timeout |
| Papers done / failed / lost | 500 / 0 / 0 |
| Requests paid for twice | 0 |
