CREATE TABLE IF NOT EXISTS doris_table (
    nom_scientifique TEXT,
    lien_doris TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (nom_scientifique)
);

CREATE TABLE IF NOT EXISTS ml_crops (
    id_crops INT PRIMARY KEY,
    id_image INT,
    url_crops VARCHAR[],
    regne TEXT,
    espece TEXT
);

CREATE TABLE IF NOT EXISTS observations_biolit_api (
    id_observation INT PRIMARY KEY,
    date_observation TIMESTAMP,
    lien_observation TEXT,
    observateur TEXT,
    url_sortie TEXT,
    espece_identifiee TEXT,
    heure_debut TIME,
    heure_fin TIME,
    latitude FLOAT,
    longitude FLOAT,
    photos TEXT,
    relais INT,
    id_espece INT,
    nom_scientifique TEXT,
    nom_commun TEXT,
    categorie_programme INT,
    programme TEXT,
    validee TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);