# Kalman Filter: Tracking a Car Around a Race Track

A Kalman filter is a **state estimation** algorithm. It is very well suited to tracking a moving car, and tracking a car around a race track is a good way to understand what a Kalman filter does and why it is useful.

The basic idea is that we have two sources of information:

1. A **motion model** -- the car cannot jump arbitrarily from one position to another. Its current position and velocity give a good indication of where it will be a short time later.

2. **Measurements** -- noisy observations of the car's position, perhaps from GPS or a camera.

The Kalman filter combines these two sources to estimate the car's actual position and velocity. Crucially, it does not simply average the measurements: it adjusts how much it trusts the measurements relative to its prediction, based on their estimated uncertainty.

## 1. Set up the problem

Start with a simplified example: the car is travelling along a straight section of track, and we observe its position every 0.1 seconds.

| Time (s) | Actual position (m) | Measured position (m) |
| :------- | :------------------ | :-------------------- |
| 0.0      | 0                   | 1.2                   |
| 0.1      | 2                   | 1.4                   |
| 0.2      | 4                   | 5.1                   |
| 0.3      | 6                   | 5.0                   |
| 0.4      | 8                   | 9.3                   |
| 0.5      | 10                  | 9.1                   |

The measured positions are noisy: sometimes they underestimate the position, sometimes they overestimate it. We want to estimate the actual position, but also the velocity and acceleration of the car.

## 2. Define the state vector

The first step is to decide what information we want to estimate. We define a state vector: $$\mathbf{x}_t = \begin{pmatrix} p_t \\ v_t \\ a_t \end{pmatrix}$$where $$p_t$$ is the position, $$v_t$$ is the velocity and $$a_t$$ is the acceleration.

Notice that acceleration is included even though we never measure it directly. This is one of the important features of a Kalman filter: it can estimate quantities that are not directly observed, provided they influence the quantities we do observe.

## 3. The motion model

We need a model describing how the state changes over time. Using elementary kinematics, assuming acceleration is approximately constant over a short interval $$\Delta t$$: $$\begin{aligned}
p_{t+1} &= p_t + v_t\Delta t + \frac{1}{2}a_t\Delta t^2 \\
v_{t+1} &= v_t + a_t\Delta t \\
a_{t+1} &= a_t
\end{aligned}$$These three equations can be written as one matrix multiplication, $$\mathbf{x}_{t+1}=\mathbf{F}\,\mathbf{x}_t$$where $$\mathbf{F}$$ is the **state transition matrix**: $$\mathbf{F} = \begin{pmatrix}
1 & \Delta t & \frac{1}{2}\Delta t^2 \\
0 & 1 & \Delta t \\
0 & 0 & 1
\end{pmatrix}$$For our example with $$\Delta t = 0.1$$: $$\mathbf{F} = \begin{pmatrix}
1 & 0.1 & 0.005 \\
0 & 1 & 0.1 \\
0 & 0 & 1
\end{pmatrix}$$There is an important qualification: the car's acceleration is variable. Assuming constant acceleration forever would be unrealistic, so we allow the acceleration to change -- which we do next.

## 4. Allow for variable acceleration

We do not actually believe acceleration remains constant. The driver might accelerate out of a corner, brake before the next corner, or maintain a steady speed.

We therefore introduce some uncertainty into the motion model, described by the **process noise covariance matrix** $$\mathbf{Q}$$.

Conceptually, $$\mathbf{Q}$$ tells the filter how much we expect the motion model to be wrong:

* **Small process noise**: acceleration changes slowly, so the motion model is fairly accurate.

* **Large process noise**: acceleration is allowed to change considerably between observations.

A useful model is to assume acceleration follows a **random walk**, $$a_{t+1}=a_t+w_t$$where $$w_t$$ is a random change in acceleration. This is natural for a car, because acceleration stays similar over a short interval but can change when the driver applies the throttle or brakes.

The key point is that we are *not* insisting the car follows our motion model exactly -- we are explicitly allowing it to deviate.

## 5. Model the observations

Suppose we only receive measurements of the car's position. The measurement vector is simply $$\mathbf{z}_t = \begin{pmatrix} p_{\text{measured}} \end{pmatrix}$$and it is related to the state by $$\mathbf{z}_t = H\mathbf{x}_t + \mathbf{r}_t$$where $$H = \begin{pmatrix} 1 & 0 & 0 \end{pmatrix}$$and $$\mathbf{r}_t$$ is measurement noise.

We also define the **measurement noise covariance matrix** $$\mathbf{R}$$. In this example it is a single number -- the variance of the position measurements. If our GPS has a standard deviation of 2 metres, $$R = 2^2 = 4$$which tells the filter how much confidence to place in the measurements.

We now have three important components:

| Component      | Purpose                                   |
| :------------- | :---------------------------------------- |
| $$\mathbf{F}$$ | Predicts how the car moves                |
| $$\mathbf{Q}$$ | Describes uncertainty in the motion model |
| $$\mathbf{R}$$ | Describes uncertainty in measurements     |

## 6. The Kalman filter algorithm

The filter operates in two alternating stages.**Stage 1 -- Prediction.** Use the previous estimate to predict the new state, $$\hat{x}_{t|t-1}=F\hat{x}_{t-1|t-1}$$and predict how uncertain that estimate is, $$P_{t|t-1}=FP_{t-1|t-1}F^T+Q$$Here $$P$$ is the covariance matrix of the state estimate. It describes the uncertainty in position, velocity and acceleration, including their correlations.**Stage 2 -- Measurement update.** Receive a new position measurement and compare it with the prediction.

First compute the prediction error (innovation), $$y_t=z_t-H\hat{x}_{t|t-1}$$then the Kalman gain, $$K_t=P_{t|t-1}H^T\left(HP_{t|t-1}H^T+R\right)^{-1}$$which determines how much we adjust the prediction in response to the new measurement. Finally update the state, $$\hat{x}_{t|t}=\hat{x}_{t|t-1}+K_t y_t$$and its uncertainty, $$P_{t|t}=\left(I-K_tH\right)P_{t|t-1}$$The interesting part is the Kalman gain:

* If measurement noise is high (large $$\mathbf{R}$$), the filter relies more on its motion prediction.

* If the motion model is uncertain (large $$\mathbf{Q}$$), the filter relies more on the measurements.

* If the measurement is particularly reliable, it pulls the estimate towards the observed position.

The filter calculates this balance automatically at every observation.

## 7. Implement the 1D filter

Let us turn the equations above into Wolfram Language code.

```wl
In[1]:=
(* Time step between observations *)
	dt = 0.1;
	
	(* State transition matrix: constant-acceleration model *)
	F = {{1, dt, dt ^ 2 / 2}, {0, 1, dt}, {0, 0, 1}};
	
	(* Measurement matrix: we observe position only *)
	H = {{1, 0, 0}};
	
	(* Measurement noise covariance: sigma = 2 m, variance = 4 *)
	R = {{4}};
	
	(* Process noise: random changes in acceleration with std dev qa *)
	qa = 0.5;
	G = {dt ^ 2 / 2, dt, 1};
	Q = qa ^ 2 Outer[Times, G, G];
	
	(* One prediction + update step of a linear Kalman filter *)
	kalmanStep[{x_, P_}, z_, F_, H_, Q_, R_] := Module[{xPred, PPred, y, S, K}, 
	  (* Prediction *)
	  xPred = F . x;
	  PPred = F . P . Transpose[F] + Q;
	  (* Measurement update *)
	  y = z - H . xPred;                     (* innovation *)
	  S = H . PPred . Transpose[H] + R;      (* innovation covariance *)
	  K = PPred . Transpose[H] . Inverse[S]; (* Kalman gain *)
	  {xPred + K . y, (IdentityMatrix[Length[x]] - K . H) . PPred}
	];
```

Let's look at the matrices:

```wl
In[9]:=
Grid[{{"F", MatrixForm[F]}, {"H", MatrixForm[H]}, {"Q", MatrixForm[Q]}, {"R", MatrixForm[R]}}, 
	 Frame -> All, Alignment -> Left]
```

Now run the filter on the table of measurements:

```wl
In[10]:=
step1D[{x_, P_}, z_] := kalmanStep[{x, P}, z, F, H, Q, R];
	
	(* Observations from the table above *)
measurements = {1.2, 1.4, 5.1, 5.0, 9.3, 9.1};
actual = {0, 2, 4, 6, 8, 10};
	
	(* Initial estimate: start at the first measurement, zero velocity/acceleration *)
x0 = {First[measurements], 0, 0};
P0 = 10 IdentityMatrix[3];
	
	(* Run the filter through the measurements *)
traj = FoldList[step1D, {x0, P0}, Rest[measurements]];
xHistory = traj[[All, 1]];
posEst = xHistory[[All, 1]];
velEst = xHistory[[All, 2]];
accEst = xHistory[[All, 3]];
	
Grid[Prepend[
	  Transpose[{Range[0, 5], actual, measurements, posEst, velEst, accEst}], 
	  {"t (s)", "Actual (m)", "Measured (m)", "Filtered (m)", "v est (m/s)", "a est (m/s^2)"}], 
	  Frame -> All]

In[21]:=
times = Range[0, 5] * dt;
ListPlot[{
	   Transpose[{times, posEst}], 
	   Transpose[{times, measurements}], 
	   Transpose[{times, actual}]}, 
	  PlotMarkers -> {Automatic, 12}, 
	  Joined -> True, 
	  PlotStyle -> {Blue, Red, Black}, 
	  PlotLegends -> {"Filtered", "Measured", "Actual"}, 
	  AxesLabel -> {"Time (s)", "Position (m)"}, 
	  ImageSize -> 500]
```

The filtered positions are pulled towards the actual positions while smoothing out the measurement noise.

## 8. Inferring velocity and acceleration from position only

Imagine the car is travelling at 20 m/s and the filter has estimated its acceleration to be zero. Its model predicts that after 0.1 s it will have travelled another 2 m. If the next GPS measurement shows 2.3 m, the filter sees a discrepancy and responds by adjusting its estimates of position, velocity *and* acceleration.

Even though we only measure position, the filter can infer that the car is accelerating because the observed positions consistently deviate from the predictions.

There is a trade-off: if the process noise is too low, the filter is slow to recognise changes in acceleration; if it is too high, it reacts strongly to measurement noise. Finding a sensible $$\mathbf{Q}$$ is an important part of tuning.

Let's demonstrate this with a longer simulation in which the car accelerates, brakes, and cruises:

```wl
In[23]:=
SeedRandom[42];
dt = 0.1;
n = 120; (* 12 seconds *)
time = Range[0, n - 1] * dt;
	
	(* True acceleration profile: accelerate, brake, cruise *)
aTrue = Table[Piecewise[{{2, 0 <= t < 4}, {-1.5, 4 <= t < 8}, {0.5, True}}], {t, time}];
	
	(* Simulate the true position and velocity *)
pTrue = ConstantArray[0, n];
vTrue = ConstantArray[0, n];
vTrue[[1]] = 10;
Do[
	  vTrue[[i]] = vTrue[[i - 1]] + aTrue[[i - 1]] * dt;
	  pTrue[[i]] = pTrue[[i - 1]] + vTrue[[i - 1]] * dt + 0.5 aTrue[[i - 1]] * dt ^ 2, 
	  {i, 2, n}
	];
	
	(* Noisy position measurements *)
sigma = 2;
measurements = pTrue + RandomVariate[NormalDistribution[0, sigma], n];
	
	(* Run the Kalman filter *)
traj = FoldList[step1D, {{First[measurements], 0, 0}, 10 IdentityMatrix[3]}, Rest[measurements]];
xHistory = traj[[All, 1]];
posEst = xHistory[[All, 1]];
velEst = xHistory[[All, 2]];
accEst = xHistory[[All, 3]];

In[39]:=
ListPlot[{Transpose[{time, posEst}], Transpose[{time, measurements}], Transpose[{time, pTrue}]}, 
	  Joined -> True, PlotStyle -> {Blue, {Red, Opacity[0.5]}, Black}, 
	  PlotLegends -> {"Filtered", "Measured", "Actual"}, 
	  AxesLabel -> {"Time (s)", "Position (m)"}, ImageSize -> 500, 
	  PlotLabel -> "Position"]
```

The velocity and acceleration are never measured, yet the filter recovers them from position alone:

```wl
In[40]:=
Grid[{{ListPlot[{Transpose[{time, velEst}], Transpose[{time, vTrue}]}, 
	      Joined -> True, PlotStyle -> {Blue, Black}, 
	      PlotLegends -> {"Filtered", "Actual"}, 
	      AxesLabel -> {"Time (s)", "Velocity (m/s)"}, ImageSize -> 400, 
	      PlotLabel -> "Velocity"], 
	    ListPlot[{Transpose[{time, accEst}], Transpose[{time, aTrue}]}, 
	      Joined -> True, PlotStyle -> {Blue, Black}, 
	      PlotLegends -> {"Filtered", "Actual"}, 
	      AxesLabel -> {"Time (s)", "Acceleration (m/s^2)"}, ImageSize -> 400, 
	      PlotLabel -> "Acceleration"]}}]
```

## 9. The 2D race track

So far we have assumed a straight line. For a real race track we extend the model to two dimensions: instead of a single position we estimate $$\mathbf{x}_t = \begin{pmatrix} x \\ y \\ v_x \\ v_y \\ a_x \\ a_y \end{pmatrix}$$applying the same constant-acceleration model independently to the two coordinate directions, with measurements $$\mathbf{z}_t = \begin{pmatrix} x_{\text{measured}} \\ y_{\text{measured}} \end{pmatrix}$$The full implementation:

```wl
In[41]:=
SeedRandom[7];
dt = 0.1;
n = 300; (* 30 seconds *)
time = Range[0, n - 1] * dt;
	
radius = 100; (* track radius *)
	
	(* Tangential speed varies: the car accelerates and brakes around the track *)
vTang = 30 + 8 Sin[2 Pi time / 20];
	
	(* Integrate the speed to get the angle around the circle *)
theta = ConstantArray[0, n];
Do[theta[[i]] = theta[[i - 1]] + vTang[[i - 1]] / radius * dt, {i, 2, n}];
	
	(* True Cartesian trajectory *)
xTrue = radius Cos[theta];
yTrue = radius Sin[theta];
	
	(* Noisy GPS measurements *)
sigma = 3;
xMeas = xTrue + RandomVariate[NormalDistribution[0, sigma], n];
yMeas = yTrue + RandomVariate[NormalDistribution[0, sigma], n];
	
	(* 2D constant-acceleration model *)
F2 = {{1, 0, dt, 0, dt ^ 2 / 2, 0}, {0, 1, 0, dt, 0, dt ^ 2 / 2}, 
	      {0, 0, 1, 0, dt, 0}, {0, 0, 0, 1, 0, dt}, 
	      {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}};
H2 = {{1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0}};
R2 = sigma ^ 2 IdentityMatrix[2];
	
qa2 = 1.0; (* std dev of random acceleration change in each axis *)
G2 = {{dt ^ 2 / 2, 0}, {0, dt ^ 2 / 2}, {dt, 0}, {0, dt}, {1, 0}, {0, 1}};
Q2 = qa2 ^ 2 (G2 . Transpose[G2]);
	
step2D[{x_, P_}, z_] := kalmanStep[{x, P}, z, F2, H2, Q2, R2];
	
	(* Run the filter over the noisy GPS positions *)
z = Transpose[{xMeas, yMeas}];
traj2 = FoldList[step2D, {{z[[1, 1]], z[[1, 2]], 0, 0, 0, 0}, 10 IdentityMatrix[6]}, Rest[z]];
xHist2 = traj2[[All, 1]];
xFilt = xHist2[[All, 1]];
yFilt = xHist2[[All, 2]];
	
Print["RMS error  measured vs true: ", RootMeanSquare[Flatten[{xMeas - xTrue, yMeas - yTrue}]]];
Print["RMS error  filtered vs true: ", RootMeanSquare[Flatten[{xFilt - xTrue, yFilt - yTrue}]]];
```

We should see three trajectories -- the actual one, the noisy measured one, and the filtered one:

```wl
In[68]:=
Show[
	  ListPlot[Transpose[{xMeas, yMeas}], PlotStyle -> {Red, PointSize[0.008], Opacity[0.6]}], 
	  ListPlot[Transpose[{xTrue, yTrue}], PlotStyle -> {Black, Thick}, Joined -> True], 
	  ListPlot[Transpose[{xFilt, yFilt}], PlotStyle -> {Blue, Thick}, Joined -> True], 
	  PlotLegends -> {"Measured", "Actual", "Filtered"}, 
	  AspectRatio -> Automatic, 
	  AxesLabel -> {"x (m)", "y (m)"}, 
	  PlotLabel -> "Kalman filter on a circular race track", 
	  ImageSize -> 500]
```

The filtered trajectory should be much closer to the actual one than the raw measurements.

## 10. Conclusion

A Kalman filter is not simply a smoothing algorithm -- it is a **state estimation** algorithm. It uses a model of how the system evolves over time to estimate hidden variables (such as velocity and acceleration) as well as directly observed variables (position).

That distinction is what makes it so useful in vehicle tracking, robotics, navigation and financial time-series modelling.