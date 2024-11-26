from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import uvicorn
import socket
import logging
import routers
import sys

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists.
        try:
            level = logger.level(record.levelname).name

            if level == 'DEBUG':
                return 0
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
        return None

logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

base_app_prefix = '/api'

app = FastAPI(
    docs_url=base_app_prefix + '/docs',
    redoc_url=base_app_prefix + '/redoc',
    openapi_url=base_app_prefix + '/openapi.json'
)
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(routers.router, prefix=base_app_prefix)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    host = '0.0.0.0'
    port = 9001

    logger.info(f'Сервис запущен ip: {host}:{port}')

    uvicorn.run(app, host=host, port=port)
