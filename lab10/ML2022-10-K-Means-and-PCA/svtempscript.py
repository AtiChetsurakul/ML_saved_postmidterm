wafer = pd.read_csv('silicon-wafer-thickness.csv')
wafer.columns

X_std = StandardScaler().fit_transform(wafer)

mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix n%s' %cov_mat)

#Calculating eigenvectors and eigenvalues on covariance matrix
cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors n%s' %eig_vecs)
print('nEigenvalues n%s' %eig_vals)

pca = PCA(n_components=N)
pca.fit_transform(wafer)
print(pca.explained_variance_ratio_)

pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()