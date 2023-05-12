from abc import ABC

import tornado

from ..predictor import FlatRegressor


class FlatRegressorApi(tornado.web.RequestHandler, ABC):
    def initialize(self, regressor: FlatRegressor, version: int) -> None:
        self.regressor = regressor
        self.version = version

    async def post(self) -> None:
        samples = tornado.escape.json_decode(self.request.body)["samples"]
        predictions = self.regressor.predict(samples)
        result = {"version": self.version, "results": predictions}
        self.write(result)
        self.set_status(200)


class Healthcheck(tornado.web.RequestHandler, ABC):
    def initialize(self, version: int) -> None:
        self.version = version

    async def get(self) -> None:
        self.write({"results": "healthy", "service": "sentiment", "version": self.version})
        self.set_status(200)
