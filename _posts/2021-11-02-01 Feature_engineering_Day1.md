
## Feature engineering Day 1 : State-space modeling and Kalman Filter

Table of contents
1. Basic concepts of state-space model and kalman filter

2. Pros and cons

3. Implementation with pykalman

## Basic concepts of state=space modeling and kalman filter

- The basic concept behind this algorithm is that we smoothe out a process by extracting its hidden state from the current state.
It adapts to new information as it arrives, so ...?

- 状態方程式から推定された状態推定値を同時期に得られた観測値によって修正する、この一連の動作をフィルタリングと呼ぶ。

## Pros and cons

Pros : 
1. It works in non-stationary data because of its dynamically changing distirbutional characteristics. 

Cons :
1. It cannot work well in a non-linear system.


## Implementation with pykalman

