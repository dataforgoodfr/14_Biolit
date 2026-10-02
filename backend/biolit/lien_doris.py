import requests
import time
from bs4 import BeautifulSoup
import polars as pl

import structlog

from biolit import DATADIR

LOGGER = structlog.get_logger()

def scrapping_site_lien_doris() -> pl.DataFrame:
    offset = 0
    lien_doris_all_data = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    print("Starting scrapping of DORIS species links...")
    while True:
        url = f"https://doris.ffessm.fr/find/species/offset/{offset}/state/*/sortby/recent/manualSort/1/view/list"
        LOGGER.info(f"Scraping offset = {offset}")

        try:
            response = requests.get(url, headers=headers, timeout=10)
            LOGGER.info(f"Status code: {response.status_code}")  # Log pour débogage

            if response.status_code != 200:
                LOGGER.error(f"Erreur HTTP {response.status_code} à l'offset : {offset}")
                break

            soup = BeautifulSoup(response.text, "html.parser")
            species = soup.find_all("div", class_="specieSearchResult resultLine")
            LOGGER.info(f"Nombre d'espèces trouvées à l'offset {offset}: {len(species)}")

            if not species:
                LOGGER.info("Fin des pages.")
                break

            lien_doris_page_data = []
            for specie in species:
                try:
                    a_tag = specie.find('a', href=True)
                    if not a_tag:
                        continue
                    lien_doris = a_tag.get("href")
                    nom_scientifique_tag = a_tag.find("em")
                    if not nom_scientifique_tag:
                        continue
                    nom_scientifique = nom_scientifique_tag.get_text(strip=True)
                    lien_doris_page_data.append({
                        "nom_scientifique": nom_scientifique,
                        "lien_doris": lien_doris,
                    })
                except Exception as e:
                    LOGGER.error(f"Erreur parsing espèce : {e}")
                    continue

            lien_doris_all_data.extend(lien_doris_page_data)
            offset += len(lien_doris_page_data)
            LOGGER.info(f"Total espèces scrapées: {len(lien_doris_all_data)}")

            df = pl.DataFrame(lien_doris_all_data)
            df.write_csv(DATADIR / "doris_data.csv")
            time.sleep(2)  # Délai augmenté

        except Exception as e:
            LOGGER.error(f"Erreur requête : {e}")
            break

    return pl.DataFrame(lien_doris_all_data)


if __name__ == "__main__":
    df = scrapping_site_lien_doris()
    print(f"Scraping terminé. {len(df)} espèces récupérées.")
    print(f"Fichier sauvegardé dans : {DATADIR / 'doris_data.csv'}")