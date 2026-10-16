"""Shared Clay patch-grid and central-pooling conventions for training and inference."""

# Clay v1.5 patch grid on our chips: 128→224 resize, patch=8 → 28×28 tokens.
PATCH_GRID = 28


def centre_pool_slice(pool_size, grid=PATCH_GRID):
    """Return (lo, hi) indices for a pool_size×pool_size window centred on the
    chip centre in a ``grid``×``grid`` patch grid.

    The chip centre falls at patch ≈ 13.9 after the 128→224 resize (chip pixel
    111.5 / patch width 8). Odd pool_sizes centre on patch 13 — matching the
    original 5×5 [11:16] convention (11+15 midpoint = 13). Even pool_sizes take
    the [13:13+pool_size] window that straddles the true centre.
    """
    if pool_size % 2 == 1:
        c = 13
        lo = c - pool_size // 2
    else:
        lo = 13
    hi = lo + pool_size
    assert 0 <= lo and hi <= grid, f"pool {pool_size} out of grid {grid}"
    return lo, hi


# Largest supported pool. The cache stores the CACHE_K × CACHE_K centre window
# once; any POOL_SIZE ≤ CACHE_K is served by a sub-slice at load time. Bumping
# this requires re-running precompute to widen the stored window.
CACHE_K = 7
CACHE_LO, CACHE_HI = centre_pool_slice(CACHE_K)  # (10, 17)


def cache_local_slice(pool_size):
    """Translate a POOL_SIZE window from full-grid indices to cache-local
    indices (0..CACHE_K). Fails loudly if POOL_SIZE > CACHE_K.
    """
    assert pool_size <= CACHE_K, (
        f"POOL_SIZE={pool_size} > CACHE_K={CACHE_K}; re-run precompute with "
        f"a wider CACHE_K to support this pool"
    )
    lo, hi = centre_pool_slice(pool_size)
    return lo - CACHE_LO, hi - CACHE_LO
