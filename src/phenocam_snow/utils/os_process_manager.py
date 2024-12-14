import multiprocessing
from typing import Any

from phenocam_snow.utils.logger import Logger


class OsProcessManager:
    @staticmethod
    def start_processes(
        target_function,
        iterables: list[Any],
        constants: list[Any],
        total_processes: int,
        logger: Logger | None = None,
    ) -> list[Any]:
        process_manager = multiprocessing.Manager()

        list_proxy = process_manager.list()
        if logger is not None:
            logger.out(f"Process manager started {total_processes} process(es).")

        processes = []
        for process_id in range(total_processes):
            slice_per_process = iterables[process_id::total_processes]
            process_args = (process_id, slice_per_process, *constants, list_proxy)
            process = multiprocessing.Process(
                target=target_function,
                args=process_args,
            )
            processes.append(process)
            process.start()

        for current_process in processes:
            current_process.join()

        if logger is not None:
            if not any(list_proxy):
                logger.out(
                    f"All processes executing the {target_function} have completed successfully."
                )
            else:
                logger.out(f"OS processes completion status: {list_proxy}")

        return list(list_proxy)
