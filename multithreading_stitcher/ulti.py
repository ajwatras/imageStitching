import numpy as np

def kmean(pts, num_cluster):
    cluster_centers = pts[:,0:num_cluster]
    pts_idx = np.zeros((1, pts.shape[1]), np.int);
    diff = 1
    while diff != 0:
        # assigning clusters
        for i in range(pts.shape[1]):
            min_idx = 0
            min_dist = np.sum(np.square(cluster_centers[:,0] - pts[:,i]))
            for j in range(1, num_cluster):
                temp = np.sum(np.square(cluster_centers[:,j] - pts[:,i]))
                if min_dist > temp:
                    min_idx = j
                    min_dist = temp
            pts_idx[0, i] = min_idx
        new_cluster = np.zeros((3, num_cluster))
        # find new cluster centers
        for i in range(pts.shape[1]):
            new_cluster[0:2, pts_idx[0, i]] = new_cluster[0:2, pts_idx[0, i]] + pts[:, i]
            new_cluster[2, pts_idx[0, i]] = new_cluster[2, pts_idx[0, i]] + 1
        new_cluster_centers = np.zeros((2, num_cluster))
        for i in range(num_cluster):
            new_cluster_centers[:, i] = new_cluster[0:2, i] / new_cluster[2, i]
        # assign new cluster centers
        diff = np.sum(np.absolute(new_cluster_centers - cluster_centers))
        cluster_centers = new_cluster_centers

    cluster_err = np.zeros((2, num_cluster))
    for i in range(pts.shape[1]):
        cluster_err[0, pts_idx[0, i]] = cluster_err[0, pts_idx[0, i]] + np.sum(np.square(pts[:, i] - cluster_centers[:, pts_idx[0, i]]))
        cluster_err[1, pts_idx[0, i]] = cluster_err[1, pts_idx[0, i]] + 1
    err = np.zeros((1, num_cluster))
    for i in range(num_cluster):
        err[0, i] = cluster_err[0, i] / cluster_err[1, i]
    return cluster_centers.astype('int'), np.sum(err)

def find_pairs(pts_main_view, pts_side_view, H):
    pts_append_1 = np.mat([[pts_side_view[1, 0], pts_side_view[1, 1]], [pts_side_view[0, 0], pts_side_view[0, 1]], [1, 1]])
    pts_append_1 = H * pts_append_1
    transformed_side_view_pts = np.zeros((2, 2))
    transformed_side_view_pts[0, 0] = pts_append_1[1, 0] / pts_append_1[2, 0]
    transformed_side_view_pts[1, 0] = pts_append_1[0, 0] / pts_append_1[2, 0]
    transformed_side_view_pts[0, 1] = pts_append_1[1, 1] / pts_append_1[2, 1]
    transformed_side_view_pts[1, 1] = pts_append_1[0, 1] / pts_append_1[2, 1]
    transformed_side_view_pts = transformed_side_view_pts.astype('int')

    diff_max = max(np.sum(np.square(pts_main_view[:, 0] - transformed_side_view_pts[:, 0])), np.sum(np.square(pts_main_view[:, 1] - transformed_side_view_pts[:, 1])))
    diff_max_reverse = max(np.sum(np.square(pts_main_view[:, 0] - transformed_side_view_pts[:, 1])), np.sum(np.square(pts_main_view[:, 1] - transformed_side_view_pts[:, 0])))

    if diff_max < diff_max_reverse:
        return pts_side_view
    else:
        return np.mat([[pts_side_view[0, 1], pts_side_view[0, 0]], [pts_side_view[1, 1], pts_side_view[1, 0]]], np.int)
