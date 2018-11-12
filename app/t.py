from math import sin, cos, pi

latlng = {
    'kor': [36.2415852, 127.5614499],
    'jap': [33.9606874, 132,5005588],
    'cha': [36.4198096, 120.4392376]
}

angle = pi / 4
cur = 0

branch = {
    'kor': [],
    'cha': [],
    'jap': []
}

for i in range(8):
    for br in latlng.keys():
        branch[br].append([sin(cur) + latlng[br][0], cos(cur) + latlng[br][1]])
    cur += angle

print(branch)
print(len(branch['kor']))
print(len(branch['cha']))
print(len(branch['jap']))