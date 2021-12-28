class Coordinate:
    """
    Cooordinate Class to define cooordinate included in its attributes
    """
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.length = y + h
        self.wide = x + w

    def get_coordinate(self):
        """[summary]
        """
        return(self.x, self.y, self.wide, self.length)


def _calculate(eye_list):
    """[summary]

    Args:
        eye_list ([type]): [description]

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    if len(eye_list) < 2:
        raise Exception("There should be 2 eyes, but have {} eyes detected".format(len(eye_list)))
    else:
        eye_1, eye_2 = eye_list[0:2]
        w3 = Coordinate(min(eye_1.x, eye_2.x), max(eye_1.y, eye_2.y), max(eye_2.wide, eye_1.wide) - min(eye_1.x, eye_2.x), max(eye_1.h, eye_2.h))
        return w3


def _get_coord(coord):
    """[summary]

    Args:
        coord ([type]): [description]

    Returns:
        [type]: [description]
    """
    return coord.get_coordinate()