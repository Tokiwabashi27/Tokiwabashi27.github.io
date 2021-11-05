
## Smoothing time-series data using Kalman filter

### 1. The intuition behind Kalman filter
The concept of Kalman filter is that extracting the true state of the observed data full of some noise and updating it as we get new information, 
we can smooth out the time-series data.

### 2. Visual understanding


```Python

#y = pd.Series(passengers['#Passengers'].values, index=pd.to_datetime(passengers['Month']
y = pd.Series(passengers['#Passengers'].values, index = pd.to_datetime(passengers['Month'],
                                                                       infer_datetime_format=True))
y = y.astype(float)
y.head()

n_train = 120
train_data, test_data = y.values[:n_train], y.values[n_train:]
train_data
test_data

n_dim_obs = 1                  # 観測値の次元数
n_dim_trend = 2                # トレンドの次元数（状態の次元数）
n_dim_state = n_dim_trend

#FとGに関しては、n_dim_trendを起点に決定されるイメージを持つ
F = np.array([
    [2, -1],
    [1, 0]
], dtype=float)

G = np.array([
    [1],
    [0]
], dtype=float)

#Hに関しては、n_dim_obsを起点に決定される
H = np.array([
    [1, 0]
], dtype=float)

Q = np.eye(1) * 10
Q = G.dot(Q).dot(G.T)

state_mean = np.zeros(n_dim_state)              # 状態の平均値ベクトルの初期値
state_cov = np.ones((n_dim_state, n_dim_state)) # 状態の分散共分散行列の初期値

#パラメータの設定ができたので状態Tretを推定して観測値tを推定する。
#状態の平均と共分散を求めれば状態の値の分布が決まる。

kf = KalmanFilter(
    n_dim_obs=n_dim_obs,
    n_dim_state=n_dim_state,
    initial_state_mean=state_mean,
    initial_state_covariance=state_cov,
    transition_matrices=F,
    transition_covariance=Q,
    observation_matrices=H,
    observation_covariance=1.0,
)

state_means, state_covs = kf.smooth(train_data)

print('状態の平均値 : \n{} \n\n 状態の共分散 : \n{}'.format(
    state_means[:5],
    state_covs[:5]
))

train_data, test_data = y[:n_train], y[n_train:]
    
state_means, state_covs = kf.smooth(train_data)
ovsevation_means_predicted = np.dot(state_means, kf.observation_matrices.T)

current_state = state_means[-1]
current_cov = state_covs[-1]

pred_means = np.array([])
for i in range(len(test_data)):

    current_state, current_cov = kf.filter_update(
        current_state, current_cov, observation=None
    )
    pred_mean = kf.observation_matrices.dot(current_state)
    pred_means = np.r_[pred_means, pred_mean]
    
plt.figure(figsize=(8, 6))
plt.plot(y.values, label="observation")
plt.plot(
    np.hstack([
        ovsevation_means_predicted.flatten(), 
        np.array(pred_means).flatten()
    ]), 
    '--', label="forecast"
)
  

```

