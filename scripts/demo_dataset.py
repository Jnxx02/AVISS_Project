import numpy as np

X = np.load("data/X_vektor_60s.npy", mmap_mode="r")
groups = np.load("data/groups_id.npy")

selected_idx = []

for uid in np.unique(groups):

    idx = np.where(groups == uid)[0]

    n_sample = min(20, len(idx))

    sampled = np.random.choice(
        idx,
        size=n_sample,
        replace=False
    )

    selected_idx.extend(sampled)

selected_idx = np.array(selected_idx)

X_demo = X[selected_idx]
groups_demo = groups[selected_idx]

np.save("data/X_demo.npy", X_demo)
np.save("data/groups_demo.npy", groups_demo)

print("X_demo:", X_demo.shape)
print("groups_demo:", groups_demo.shape)
print("ID unik:", np.unique(groups_demo))