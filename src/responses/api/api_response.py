class ApiResponse:

    def __init__(self, data):
        self.data = data

    def true_output(self):
        return {"status": "success",
                "message": "car meta-tags processed successfully",
                "data": self.data}

    def error_output(self):
        return {"code": 400,
                "status": "error",
                "message": "no image detected",
                "data": []}

    def error_req(self):
        return {"code": 404,
                "status": "error",
                "message": "bad request"}
