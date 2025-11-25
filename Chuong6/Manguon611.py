kmeans_kwargs = {
    'init': 'k-means++',
    'n_init': 10,
    'max_iter': 300,
    'random_state': 42,
}
kmeans = KMeans(n_clusters=5, **kmeans_kwargs)
kmeans_labels = kmeans.fit(X_train_raw[lgbm_top_20feats[:2]]).predict(X_train_raw[lgbm_top_20feats[:2]])
scatter_2d_with_labels(X_train_raw[lgbm_top_20feats[:2]].to_numpy(), labels=kmeans_labels)
