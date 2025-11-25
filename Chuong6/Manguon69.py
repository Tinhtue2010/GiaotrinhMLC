model_fi = lgb.LGBMClassifier()
model_fi.fit(
    X_train_raw, y_train
)

lgbm_scores = model_fi.feature_importances_

lgbm_feats_score = list(zip(X_train_raw.columns, lgbm_scores))
lgbm_feats_score.sort(key=lambda x: -x[1])

plot_feature_importance(
    [i[1] for i in lgbm_feats_score][:50],
    [i[0] for i in lgbm_feats_score][:50],
    'LGBM'
)
