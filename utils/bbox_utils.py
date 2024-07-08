def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int((y1+y2)/2)


def get_bbox_width(bbox):
    return bbox[2]-bbox[0]


# measuring distance between two points for nearest feet
"""
Here p1[0] and p2[0] are the x-coordinates, p1[1] and p2[1] are the y-coordinates.

The function uses the Euclidean distance formula:
distance = √[(x2 - x1)^2 + (y2 - y1)^2]

(p1[0]-p2[0])**2: Calculates the squared difference of x-coordinates
(p1[1]-p2[1])**2: Calculates the squared difference of y-coordinates
The results are added together AND
multiplied by 0.5 is used to calculate the square root of the sum.
"""


def measure_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def measure_xy_distance(p1, p2):
    return p1[0]-p2[0], p1[1]-p2[1]


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1+x2)/2), int(y2)
