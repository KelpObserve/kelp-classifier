"""Current 50 m observation, chip-centre and duplicate-array grouping."""
import numpy as np
from sklearn.neighbors import BallTree

def group_rows(f):
    f = f.reset_index(drop=True)
    parent = list(range(len(f)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)
    for col in ['observation_id', 'chip_id', 'sha256']:
        for _, rr in f.groupby(col):
            ii = rr.index.tolist()
            for j in ii[1:]:
                union(ii[0], j)
    for lat, lon in [('decimalLatitude', 'decimalLongitude'), ('chip_center_lat', 'chip_center_lon')]:
        xy = np.radians(f[[lat, lon]].to_numpy())
        near = BallTree(xy, metric='haversine').query_radius(xy, r=0.05 / 6371.0088)
        for i, ns in enumerate(near):
            for j in ns:
                union(i, int(j))
    return np.array([find(i) for i in range(len(f))])
