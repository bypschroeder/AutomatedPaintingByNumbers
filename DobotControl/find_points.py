

def calcPoints(points):
    output = []
    for point in points:
        print("Calc Input:", point)
        new_point = []
        new_point.append(point[1] + 200)
        new_point.append(point[0] - 50)
        output.append(new_point)
    return(output)

