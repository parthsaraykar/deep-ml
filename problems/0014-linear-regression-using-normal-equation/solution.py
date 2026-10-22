import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X=np.array(X)
	y=np.array(y).reshape(-1,1)
	X_T=X.T
	theta=np.linalg.inv((X_T.dot(X))).dot(X_T).dot(y)
	theta =np.round(theta,4).flatten().tolist()

	

	return theta