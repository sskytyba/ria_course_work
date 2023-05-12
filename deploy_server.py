import logging

import tornado.ioloop
import tornado.web
from tornado.httpclient import AsyncHTTPClient

from src.deploy.api import Healthcheck, FlatRegressorApi
from src.predictor import FlatRegressor

logging.getLogger("transformers").setLevel(logging.ERROR)


MODEL_PATH = "model/"
PORT = 1492


def make_app(regressor: FlatRegressor, version: int) -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/flat_regressor/healthcheck", Healthcheck, dict(version=version)),
            (r"/flat_regressor/", FlatRegressorApi, dict(regressor=regressor, version=version)),
        ]
    )


if __name__ == "__main__":
    AsyncHTTPClient.configure("tornado.simple_httpclient.SimpleAsyncHTTPClient", max_clients=100)
    http_client = AsyncHTTPClient()
    model_version = int(open("model.version").readline().rstrip())
    regressor = FlatRegressor.load(MODEL_PATH)
    app = make_app(regressor=regressor, version=model_version)
    app.listen(PORT)
    tornado.ioloop.IOLoop.current().start()
