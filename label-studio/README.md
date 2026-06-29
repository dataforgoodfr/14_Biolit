# Label Studio

Ce dossier contient uniquement les interfaces versionnées de production :

- `config/crop.xml` pour le crop humain si YOLOv8 ne détecte rien ;
- `config/validation.xml` pour valider, corriger ou rejeter la classification.

Le backend monte ce dossier en lecture seule et crée automatiquement les deux
projets `Biolit - Crop manuel` et `Biolit - Validation taxonomique`.

Au premier démarrage, ouvrir <http://localhost:8089>, se connecter avec les
identifiants du `.env`, copier le jeton API du compte dans
`LABEL_STUDIO_API_KEY`, puis démarrer le service `backend`.

Les tâches terminées sont supprimées après l'enregistrement de leur résultat
dans PostgreSQL. L'audit durable se trouve dans `validated_species`.
