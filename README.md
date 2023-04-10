## KeyPaper

- Relation-Based Associative Joint Location for Human Pose Estimation in Videos
- Invariant Teacher and Equivariant Student for Unsupervised 3D Human Pose Estimation

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/JRE3.png?raw=true">
  <br>
  그림 1.JRE
</p>

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/ITE/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202023-02-01%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%205.14.26.png?raw=true">
  <br>
  그림 2.ITE
</p>

**GOAL** : 해당 두 모델을 End to End 로 이어 Unsupervised 3D pose estimation for invisible Joint with 2D image 

## 진행상황

### Experiment

Dataset : Penn Action data(Only)

- Case 1 : 2D joint Heatmap(100% 데이터 활용, 100 Epoch)

​	전반적으로 탐지가 잘 되나, 오차가 있음. 

- Case 2 : Heatmap to Coordinate(100% 데이터 활용)

​    모델의 output 의 Confidence 가 높으면 잘 예측하나, 그렇지 않은 경우 예측값이 벗어나는 경향이 있음 

- Case 3 : Coordinate(Ground Truth) to 3D pose estimation(모든 joint 가 Visible 인 데이터 활용 , 8 X 5 Batch 20 Iteration per 1 Epoch, 100 Epoch)

​	Heatmap 에서 Softargmax를 통해 계산된 Coord 가 아닌, 기존 데이터의 Key point 를 활용한 결과 어느정도 형태가 나오는 것을 확인

- Case 4 : Coordinate( JRE Inference Heatmap With Soft Argmax, Case3 과 동일 데이터셋, 100 Epoch) to 3D pose estimation

​	모든 joint 가 Visible 한 데이터셋을 가져와서 전체 데이터셋에 대해 Pretrained 된 JRE 모델을 Freeze 시킨 뒤 3D joint 학습(모두 Visible Joint이기 때문에 Inference 결과가 나쁘지 않음)

- Case 5 : Only Invisible Joint 2D estimation(Invisible Dataset)

​	Joint 가 Invisible인 Data에 대해 추론이 잘 되지 않는 것 같다는 문제의식이 있어 Invisible한 Joint 가 있는 경우에 대해서만 JRE Model 학습 진행

- Case 6 : 3D pose estimation (Invisible Data 포함 )

​	미진행

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/exp_1.png?raw=true">
  <br>
  그림 3. EXP Ground Truth Heatmap
  </p>

<p align = "center">
  <img width = "600" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/exp2.png?raw=true">
  <br>
  그림 4. EXP train
  </p>

<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/exp_img.png?raw=true">
  <br>
  그림 5. EXP img
  </p>



<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/exp_2d.png?raw=true">
  <br>
  그림 6. EXP 2d
  </p>





<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/exp3d_pose.png?raw=true">
  <br>
  그림 7. EXP 3d
  </p>



<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/exp_3d.png?raw=true">
  <br>
  그림 8. 3D Inference Freeze Loss
  </p>



<p align = "center">
  <img width = "400" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/exp_3dground.png?raw=true">
  <br>
  그림 9. 3D Ground Truth Loss
  </p>



- Skeleton 의 정규화를 다시 해결해야 할 거 같음

### SoftArgmax 의 문제점 & Custom (구현 완료,  실험 미진행)

invisible 한 Heatmap 의 경우 대부분의 확률이 동등하기 때문에 Joint to coor 의 문제가 해결되지 않음 (Soft Argmax 는 각 행에대한 평균값을 구하기 때문에 0으로 취급되는 Joint 에 대해 0 값을 뽑지 않고 오차가 심하게 값을 뽑음)

확률의 output 이 비슷할 때 Penalty 를 줄 수 있는가?

또한 이후 3D 로 옮길 때 0으로 나온 값들에 대해 어떻게 학습에 참여시킬 수 있는가?

불확실한 분포에서 무슨 페널티를 주는 것이 적절한가? -> Unseen 이라 취급하여 0 으로 수렴하게 하는 것이 목표

-> heatmap 에 대해 추가적인 학습 필요하다고 판단

- Heatmap 을 Convolution Layer 에 통과시켜 0 ~ 2 사이의 값을 갖게한다
  - 64 X 64 Heatmap 을 Resnet Layer 에 통과 -> GAP

- 이후 해당 값을 실제 Joint 에 곱하여 실제 Joint 와의 값과의 차이를 MSELoss를 통해 구하여 CNN 파라미터 업데이트  -> 17 X 2 의 행렬이 나올 수 있도록 설계
- 실험 설계 : 우선 Pretrained 된 JRE 의 Output 을 통해 Coord 를 구한 후 Ground Truth Keypoint와 MSE Loss를 구하여 Update

<p align = "center">
  <img width = "200" src = "https://github.com/skdytpq/skdytpq.github.io/blob/master/_pics/jre/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202023-04-05%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2010.27.02.png?raw=true">
  <br>
  그림 10. Invisible Joint image
  </p>

### Model Custom(Invisible Joint 생성)

모델에서 특정 부분에 대해 Random 하게 마스킹을 진행한 뒤 복원 과정에서 찾을 수 있게 한다?

Human 3.6M Dataset의 경우 Invisible Joint 가 없기 때문에 모델이 온전하게 모든 Joint 에 대해 학습할 수 있음

그러나 Penn Action Dataset 의 경우 Invisible Joint 가 존재하며, 몇 부분의 관절만 이미지에 담긴 데이터가 존재한다.

따라서 3D pose estimation 시 Joint 가 없는 부분을 주변 Joint 정보를 활용하여 생성할 수 있는 모델 계획

- Setting :
  1. 모든 Joint 가 Visible 이거나, Joint 추론값 중 0이 없는 Image Data 준비
  2. 해당 Image 에 각 Joint 를 Random 하게 Masking 하여 GCN Network(새로운 Module 설계) 에 투입
  3. 해당 Sub Module 을 Freeze 하여 JRE  - ITE 를 이을 때 sub information 으로 투입
