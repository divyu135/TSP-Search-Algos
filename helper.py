from math import sin, cos, sqrt, atan2, radians

def calculate_cost(path,df):
    total = 0
    for i in range(len(path)-1):
        lat1 = df['latitude'][i]
        lon1 = df['longitude'][i]
        lat2 = df['latitude'][i+1]
        lon2 = df['longitude'][i+1]

        R = 6373.0 #radius if earth

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c

        total += round(distance,3)

    return total
