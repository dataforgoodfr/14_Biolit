# Metabase

Metabase conserve sa base applicative dans `/metabase-data`. Le Compose permet
de rattacher directement un volume Docker déjà restauré sur la machine :

```dotenv
METABASE_VOLUME_NAME=nom_du_volume_existant
METABASE_VOLUME_EXTERNAL=true
```

Avec `METABASE_VOLUME_EXTERNAL=false`, Docker crée le volume s'il n'existe pas.
Avec `true`, le démarrage échoue volontairement si le volume indiqué est absent,
ce qui évite de lancer par erreur un Metabase vide.

Si la source BioLit n'existe pas encore dans ce Metabase, ajouter PostgreSQL :

- hôte : `postgres` ;
- port : `5432` ;
- base, utilisateur et mot de passe : valeurs du `.env` ;
- vue métier : `metabase_observations`.

Aucun fichier Metabase n'est dupliqué dans le dépôt : le volume reste son unique
persistance.
