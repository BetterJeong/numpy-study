# https://u-n-joe.tistory.com/94
# numpy import와 버전, 구성 확인
import numpy as np

print("numpy 버전: " + np.__version__)
np.show_config()

# 사이즈 10의 Null 벡터
Z = np.zeros(10)
print(Z)

# 모든 배열의 메모리 크기 찾기
Z = np.zeros((10, 10))
print("%d bytes" % (Z.size * Z.itemsize))

# 크기 10의 Null 벡터를 생성하고 다섯 번째 값은 1 출력
Z = np.zeros(10)
Z[4] = 1
print(Z)

# 값이 10 ~ 49인 벡터 만들기
Z = np.arange(10, 50)
print(Z)

# 벡터 반전
Z = np.arange(50)
Z = Z[::-1]
print(Z)

# 0 ~ 8 범위의 값 사용해서 3x3 매트릭스 만들기
Z = np.arange(9).reshape(3, 3)
print(Z)

# [1,2,0,0,4,0]에서 0이 아닌 요소의 색인 찾기
Z = np.nonzero([1, 2, 0, 0, 4, 0])
print(Z)

# 3x3 매트릭스 생성
Z = np.eye(3)
print(Z)