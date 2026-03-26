import requests
import time
from bs4 import BeautifulSoup
import polars as pl

import structlog

from biolit import DATADIR

LOGGER = structlog.get_logger()


def scrapping_site_lien_doris(max_offset: int = 100) -> pl.DataFrame:
    offset = 0
    lien_doris_all_data = []

    while True:
        if offset >= max_offset:
            LOGGER.info("Atteint le max_offset pour test, fin du scraping.")
            break
        url = f"https://doris.ffessm.fr/find/species/(offset)/{offset}/(state)/*/(sortby)/recent/(manualSort)/1/(view)/list"
        LOGGER.info(f"Scraping offset = {offset}")

        try:
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                LOGGER.info(f"Erreur à l'offset : {offset}")
                break

            soup = BeautifulSoup(response.text, "html.parser")
            species = soup.find_all("div", class_="specieSearchResult resultLine")
            if not species:
                LOGGER.info("Fin des pages.")
                break

            lien_doris_page_data = []
            for specie in species:
                try:
                    a_tag = specie.find('a', href=True)
                    lien_doris = a_tag.get("href")
                    nom_scientifique = a_tag.find("em").get_text(strip=True)
                    lien_doris_page_data.append({
                        "nom_scientifique": nom_scientifique,
                        "lien_doris": lien_doris,
                    })
                except Exception as e:
                    LOGGER.info(f"Erreur parsing espèce : {e}")
                    continue
            lien_doris_all_data.extend(lien_doris_page_data)
            offset += len(lien_doris_page_data)

            df = pl.DataFrame(lien_doris_all_data)
            df.write_csv(DATADIR / "doris_data.csv")
            time.sleep(1)

        except Exception as e:
            LOGGER.info(f"Erreur requête : {e}")
            break

    return pl.DataFrame(lien_doris_all_data)
