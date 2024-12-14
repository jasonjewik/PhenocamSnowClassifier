import logging
import re
from multiprocessing.managers import ListProxy

import requests

from phenocam_snow.utils.constants import CONFIG
from phenocam_snow.utils.logger import Logger
from phenocam_snow.utils.os_process_manager import OsProcessManager


class PhenoCamAPI:

    def __init__(self, site_name: str):
        self._site_name = site_name
        self._log_level = logging.getLevelName(CONFIG["PhenoCamAPI"]["LogLevel"])
        self._logger = Logger(name=__name__, level=self._log_level)

    @staticmethod
    def get_site_names() -> list[str]:
        """Gets PhenoCam site names.

        :return: The list of PhenoCam site names
        :rtype: list[str]

        :raises requests.exceptions.Timeout: if the GET request times out
        :raises RuntimeError: if non-200 response is received
        """
        url = "https://phenocam.nau.edu/webcam/network/table/"
        pattern = re.compile(r'\/webcam\/sites\/(.+)/"')
        resp = requests.get(url, timeout=3)
        if resp.status_code != 200:
            raise RuntimeError(f"did not get 200 response from {url}")
        return pattern.findall(resp.text)

    def get_site_months(self) -> list[str]:
        """Gets months for a PhenoCam site.

        :return: the list of URLs for the given site's months
        :rtype: list[str]

        :raises requests.exceptions.Timeout: if the GET request times out
        :raises RuntimeError: if non-200 response is received
        """
        url = f"https://phenocam.nau.edu/webcam/browse/{self._site_name}"
        pattern = re.compile(rf'\/webcam\/browse\/{self._site_name}(\/.+\/.+)"')
        resp = requests.get(url, timeout=3)
        if resp.status_code != 200:
            raise RuntimeError(f"did not get 200 response from {url}")
        return [f"{url}{path}" for path in pattern.findall(resp.text)]

    def get_site_dates(self, year: str, month: str) -> list[str]:
        """Gets dates for a PhenoCam site in a given year and month.

        :param year: The year to retrieve dates for as a string like "YYYY".
        :type year: str
        :param month: The month to retrieve dates for as a zero-leading string like "01" or "12".
        :type month: str

        :return: the list of URLs for the given site's dates in the given year and month
        :rtype: list[str]

        :raises requests.exceptions.Timeout: if the GET request times out
        :raises RuntimeError: if non-200 response is received
        """
        url = f"https://phenocam.nau.edu/webcam/browse/{self._site_name}/{year}/{month}"
        pattern = re.compile(rf'\/webcam\/browse\/{self._site_name}\/.+\/.+(\/.+)\/"')
        resp = requests.get(url, timeout=3)
        if resp.status_code != 200:
            raise RuntimeError(f"did not get 200 response from {url}")
        return [f"{url}{path}" for path in pattern.findall(resp.text)]

    def get_site_images(self, year: str, month: str, date: str) -> list[str]:
        """Gets images for a PhenoCam site in a given year, month, and date.

        :param year: The year to retrieve dates for as a string like "YYYY".
        :type year: str
        :param month: The month to retrieve dates for as a zero-leading string like "01" or "12".
        :type month: str
        :param date: The date to retrieve dates for as a zero-leading string like "01" or "30".
        :type date: str

        :return: the list of URLs for the given site's dates in the given year and month
        :rtype: list[str]

        :raises requests.exceptions.Timeout: if the GET request times out
        :raises RuntimeError: if non-200 response is received
        """
        base = "https://phenocam.nau.edu"
        url = f"{base}/webcam/browse/{self._site_name}/{year}/{month}/{date}"
        pattern = re.compile(
            rf'(\/data\/archive\/{self._site_name}\/{year}\/{month}\/.*\.jpg)"'
        )
        resp = requests.get(url, timeout=3)
        if resp.status_code != 200:
            raise RuntimeError(f"did not get 200 response from {url}")
        return [f"{base}{path}" for path in pattern.findall(resp.text)]

    def get_all_urls(self) -> list[str]:  # pragma: no cover
        """Gets all available images for a PhenoCam site.

        :return: the list of URLs for the given site's images
        :rtype: list[str]

        :raises requests.exceptions.Timeout: if raised by get_site_months
        :raises RuntimeError: if raised by get_site_months
        """
        self._logger.out(f"Retrieving all URLs for {self._site_name}")
        site_month_urls = self.get_site_months()
        self._logger.out(f"Retrieved {len(site_month_urls)} months")
        return OsProcessManager.start_processes(
            target_function=self._get_all_urls_worker,
            iterables=site_month_urls,
            constants=(int(CONFIG["PhenoCamAPI"]["GetAllUrlsUpdateFrequency"]),),
            total_processes=int(CONFIG["PhenoCamAPI"]["NumDownloadWorkers"]),
            logger=self._logger,
        )

    def _get_all_urls_worker(
        self,
        process_id: int,
        month_urls: list[str],
        update_frequency: int,
        list_proxy: ListProxy,
    ) -> None:  # pragma: no cover
        logger = Logger(name=f"get_all_urls_worker_{process_id}", level=self._log_level)
        logger.out(
            f"Process #{process_id} assigned {len(month_urls)} months to download"
        )
        year_month_str = rf"https:\/\/phenocam.nau.edu\/webcam\/browse\/{self._site_name}\/(?P<year>.+)\/(?P<month>.+)"
        year_month_pattern = re.compile(year_month_str)
        date_pattern = re.compile(year_month_str + r"\/(?P<date>.+)")
        for i, month_url in enumerate(month_urls):
            if i > 0 and i % update_frequency == 0:
                logger.out(
                    f"Process #{process_id} completed {i}/{len(month_urls)} months"
                )
            match = year_month_pattern.match(month_url)
            year, month = match.group("year"), match.group("month")
            try:
                site_date_urls = self.get_site_dates(year, month)
            except Exception as exc:
                logger.error(exc)
            for date_url in site_date_urls:
                match = date_pattern.match(date_url)
                date = match.group("date")
                try:
                    site_image_urls = self.get_site_images(year, month, date)
                except Exception as exc:
                    logger.error(exc)
                list_proxy.extend(site_image_urls)
