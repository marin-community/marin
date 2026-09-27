hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1','l2'],
    "num_iterations": 1000, 
    'seed': 42,
    'learning_rate': 1e-2,
    "verbosity": -1,
}

        
np.random.seed(42)

predictor = []

for i in range(len(KEY_METRICS)):

    target = y_train[:, i]
    test_target = y_test[:, i]
    
    gbm = lgb.LGBMRegressor(**hyper_params)

    reg = gbm.fit(X_train, target,
        eval_set=[(X_test, test_target)],
        eval_metric='l2', callbacks=[
        lgb.early_stopping(stopping_rounds=3, verbose=False),
    ])
    r, p = spearmanr(reg.predict(X_test), test_target)
    print(i, KEY_METRICS[i], "Correlation: {}".format(np.round(r*100, 2)))

    predictor.append(reg)
    # break