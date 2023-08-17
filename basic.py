# https://u-n-joe.tistory.com/94
# numpy import와 버전, 구성 확인
import numpy as np


print("numpy 버전: " + np.__version__)
np.show_config()


print("# 사이즈 10의 Null 벡터")
Z = np.zeros(10)
print(Z)


print("# 모든 배열의 메모리 크기 찾기")
Z = np.zeros((10, 10))
print("%d bytes" % (Z.size * Z.itemsize))


print("# 크기 10의 Null 벡터를 생성하고 다섯 번째 값은 1 출력")
Z = np.zeros(10)
Z[4] = 1
print(Z)


print("# 값이 10 ~ 49인 벡터 만들기")
Z = np.arange(10, 50)
print(Z)


print("# 벡터 반전")
Z = np.arange(50)
Z = Z[::-1]
print(Z)


print("# 0 ~ 8 범위의 값 사용해서 3x3 매트릭스 만들기")
Z = np.arange(9).reshape(3, 3)
print(Z)


print("# [1,2,0,0,4,0]에서 0이 아닌 요소의 색인 찾기")
Z = np.nonzero([1, 2, 0, 0, 4, 0])
print(Z)


print("# 3x3 매트릭스 생성")
Z = np.eye(3)
print(Z)


print("# 랜덤 값이 있는 3x3x3 어레이 생성")
Z = np.random.random((3, 3, 3))
print(Z)


print("# 랜덤 값이 있는 10x10 배열을 만들고 최소값, 최대값 찾기")
Z = np.random.random((10, 10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)


print("# 30 크기의 랜덤 벡터를 만들고 평균 값 찾기")
Z = np.random.random(30)
m = Z.mean()
print(m)


print("# 경계가 1이고 내부가 0인 2차원 배열 만들기")
Z = np.ones((10, 10))
Z[1:-1, 1:-1] = 0
print(Z)


print("# 기존 어레이에 테두리를 0으로 채우기")
print("방법 1:")
Z = np.ones((5, 5))
Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
print(Z)
print("방법 2:")
Z[:, [0, -1]] = 0
Z[[0, -1], :] = 0
print(Z)


print("# nan 다루기")
# inf == 무한대, nan == NaN
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3 * 0.1)


print("# 8x8 매트릭스를 만들어 체크보드 패턴으로 채우기")
Z = np.zeros((8, 8), dtype=int)
Z[1::2, ::2] = 1
Z[::2, 1::2] = 1
print(Z)


print("# 형상 배열 (6, 7, 8)을 고려한 100번째 원소의 색인(x, y, z)")
print(np.unravel_index(99, (6, 7, 8)))


print("# 타일 함수를 사용하여 체크보드 8x8 매트릭스 만들기")
Z = np.tile(np.array([[0, 1], [1, 0]]), (4, 4))
print(Z)


print("# 5x5 랜덤 행렬 정규화")
Z = np.random.random((5, 5))
Z = (Z - np.mean(Z)) / (np.std(Z))
print(Z)


# numpy 버전 문제로 실행 안 됨
# print("# 색상을 4바이트(RGBA)로 나타내는 사용자 지정 dtype 생성")
# color = np.dtype([("r", np.ubyte, 1),
#                   ("g", np.ubyte, 1),
#                   ("b", np.ubyte, 1),
#                   ("a", np.ubyte, 1)])


print("# 5x3 매트릭스에 3x2 매트릭스 곱하기")
print("방법 1.")
Z = np.dot(np.ones((5, 3)), np.ones((3, 2)))
print(Z)
print("방법 2.")
Z = np.ones((5, 3)) @ np.ones((3, 2))
print(Z)


print("# 1차원 배열이 주어지면 3과 8 사이의 모든 요소 부정하기")
Z = np.arange(11)
Z[(3 < Z) & (Z < 8)] *= -1
print(Z)


Z = None
print("# 실수형 배열 0에서 반올림")
Z = np.random.uniform(-10, +10, 10)
print("방법 1.")
print(np.copysign(np.ceil(np.abs(Z)), Z))
print("방법 2.")
print(np.where(Z > 0, np.ceil(Z), np.floor(Z)))


print("# 두 어레이 간의 공통 값 찾기")
Z1 = np.random.randint(0, 10, 10)
Z2 = np.random.randint(0, 10, 10)
print(np.intersect1d(Z1, Z2))


print("# 어제, 오늘, 내일 날짜 알기")
yesterday = np.datetime64('today') - np.timedelta64(1)
today = np.datetime64('today')
tomorrow = np.datetime64('today') + np.timedelta64(1)
print(yesterday)
print(today)
print(tomorrow)


print("# 2023년 8월에 해당하는 모든 날짜 알기")
Z = np.arange('2023-08', '2023-09', dtype='datetime64[D]')
print(Z)


print("# 복사본 없이(제자리) (A+B)*(-A/2) 계산하기")
A = np.ones(3) * 1
B = np.ones(3) * 2
C = np.ones(3) * 3
np.add(A, B, out=B)
np.divide(A, 2, out=A)
np.negative(A, out=A)
print(np.multiply(A, B, out=A))


print("# 4가지 다른 방법을 사용하여 양수 배열의 정수 부분 추출")
Z = np.random.uniform(0, 10, 10)
print(Z - Z % 1)
print(Z // 1)
print(np.floor(Z))
print(Z.astype(int))
print(np.trunc(Z))


print("# 행 값이 0~4인 5x5 매트릭스 만들기")
Z = np.zeros((5, 5))
Z += np.arange(5)
print(Z)


print("# 10개의 정수를 생성하고 이를 사용해서 배열을 만드는 함수 구현")
def generate():
    for x in range(10):
        yield x


Z = np.fromiter(generate(), dtype=float, count=-1)
print(Z)


print("# 0에서 1사이의 값으로 크기 10의 벡터 만들기")
Z = np.linspace(0, 1, 11, endpoint=False)[1:]
print(Z)


print("# 크기 10의 랜덤 벡터 만들어 정렬")
Z = np.random.random(10)
Z.sort()
print(Z)


print("# np.sum 보다 1차원 배열 빠르게 합하기")
Z = np.arange(10)
print(np.add.reduce(Z))


print("# 랜덤 배열 A B 같은지 확인")
A = np.random.randint(0, 2, 5)
B = np.random.randint(0, 2, 5)
print("방법 1.")
print(np.allclose(A, B))
print("방법 2.")
print(np.array_equal(A, B))


print("# 읽기 전용 배열")
Z = np.zeros(10)
Z.flags.writeable = False
# Z[0] = 1


print("# 데카르트 좌표를 나타내는 랜덤 10x2 행렬을 극좌표로 변환")
Z = np. random.random((10, 2))
X, Y = Z[:, 0], Z[:, 1]
R = np.sqrt(X**2 + Y**2)
T = np.arctan2(Y, X)
print(R)
print(T)


print("# 크기 10의 랜덤 벡터를 만들고 최대값 0으로 바꾸기")
Z = np.random.random(10)
Z[Z.argmax()] = 10
print(Z)


print("# [0,1]x[0,1] 영역을 포함하는 x 및 y 좌표로 구성된 배열 만들기")
Z = np.zeros((5, 5), [('x', float), ('y', float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
print(Z)


print("# X와 Y의 두 배열이 주어지면 Cauchy 행렬 C(Cij = 1/(xi - yj) 구성하기")
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))


print("# numpy 스칼라 유형에 대해 표현 가능한 최소값과 최대값 인쇄")
for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)
