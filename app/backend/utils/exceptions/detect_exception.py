class NotFoundFaceImageError(Exception):
    def __init__(self, message="There was not face found"):
        self.message = message
        super().__init__(message)
