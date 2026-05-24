# memorize bench — embed batch sweep

**chunk_chars:** 800
**target chunks per size:** 512

| batch | calls | chunks | wall ms | µs/chunk | chunks/s |
|---|---|---|---|---|---|
| 1 | 512 | 512 | 4446 | 8683.9 | 115 |
| 4 | 128 | 512 | 3750 | 7325.3 | 137 |
| 16 | 32 | 512 | 3574 | 6982.2 | 143 |
| 64 | 8 | 512 | 3675 | 7178.0 | 139 |
| 256 | 2 | 512 | 3786 | 7396.2 | 135 |

**speedup vs batch=1:**

| batch | speedup |
|---|---|
| 1 | 1.00× |
| 4 | 1.19× |
| 16 | 1.24× |
| 64 | 1.21× |
| 256 | 1.17× |

