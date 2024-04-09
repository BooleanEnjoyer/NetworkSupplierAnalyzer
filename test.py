#
# for train_index, test_index in rkf.split(X_scaled):
#     X_train_cv, X_test_cv = X_scaled[train_index], X_scaled[test_index]
#     y_train_cv, y_test_cv = y[train_index], y[test_index]
#
#     # Initialize and fit the model
#     model = LinearRegression()
#     model.fit(X_train_cv, y_train_cv)
#
#     # Evaluate the model on the test fold
#     cv_score = model.score(X_test_cv, y_test_cv)
#     cv_scores.append(cv_score)
#
# # Compute the mean and standard deviation of cross-validation scores
# mean_cv_score = sum(cv_scores) / num_folds
# std_cv_score = np.std(cv_scores)
#
# print(f'Cross-Validation Mean Score: {mean_cv_score}')
# print(f'cv_scores lenght: {len(cv_scores)}')
# print(f'Cross-Validation Mean Score: {np.mean(cv_scores)}')
# print(f'Cross-Validation Standard Deviation: {std_cv_score}')