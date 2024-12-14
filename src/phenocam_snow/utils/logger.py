import logging


class Logger:
    def __init__(self, name: str, level: int, file_name: str | None = None):
        logging.getLogger(name).setLevel(level)
        logging_format = "%(asctime)s [%(levelname)s] %(message)s"
        if file_name is not None:
            logging.basicConfig(
                level=logging.INFO, format=logging_format, filename=file_name
            )
        else:
            logging.basicConfig(level=logging.INFO, format=logging_format)

    @staticmethod
    def out(message):
        logging.info(message)

    @staticmethod
    def error(ex: Exception):
        template = "An exception of type {0} occurred. Arguments: {1!r}"
        message = template.format(type(ex).__name__, ex.args)
        logging.error(message)
