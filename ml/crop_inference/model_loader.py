from pathlib import Path
from huggingface_hub import hf_hub_download

def load_model_weights(cfg: dict) -> str:
    """Retourne le chemin local vers best.pt, quelle que soit la source."""
    source = cfg["model"]["source"]
    model_cfg = cfg["model"]
    
    # Charger la source huggingface
    if source == "huggingface":
        return hf_hub_download(
            repo_id=model_cfg["repo_id"],
            filename=model_cfg["filename"]
        )
    
    # Charger la source en local
    if source == "local":
        path = Path(model_cfg["path"])
        if not path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {path}")
        return str(path)
    
    # TODO: si besoin ajouter une source modèle sur S3 -> voir dès que le S3 est dispo

    raise ValueError(f"Source inconnue : {source!r}. Valeurs acceptées : huggingface, local.")
