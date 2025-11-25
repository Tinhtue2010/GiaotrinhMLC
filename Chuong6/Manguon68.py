from sklearn.feature_selection import f_classif
anova_scores, _ = f_classif(np.abs(X_train_raw.values), y_train)

anova_feats_score = list(zip(X_train_raw.columns, anova_scores))
anova_feats_score.sort(key=lambda x: -x[1])

plot_feature_importance(
    [i[1] for i in anova_feats_score],
    [i[0] for i in anova_feats_score],
    'Anova'
)
