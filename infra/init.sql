CREATE TABLE IF NOT EXISTS doris_table (
    nom_scientifique TEXT,
    lien_doris TEXT,
    UNIQUE (nom_scientifique)
);

CREATE TABLE IF NOT EXISTS ml_no_crops (
    run_name TEXT,
    id_observation TEXT PRIMARY KEY,
    path_s3 TEXT
);

CREATE TABLE IF NOT EXISTS ml_crops (
    run_name TEXT,
    id_crops TEXT PRIMARY KEY,
    regne TEXT,
    confiance FLOAT,
    path_s3 TEXT
);

CREATE TABLE IF NOT EXISTS ml_taxonomy (
    run_name TEXT,
    id_crops TEXT PRIMARY KEY,
    id_observation INT,
    latitude FLOAT,
    longitude FLOAT,
    regne TEXT,
    confiance FLOAT,
    path_s3 TEXT,
    nom_scientifique TEXT,
    lien_doris TEXT
);

CREATE TABLE IF NOT EXISTS observations (
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
    validee TEXT
);

CREATE TABLE IF NOT EXISTS enriched_observations_metabase (
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
    reg_nom TEXT,
    dep_nom TEXT,
    nom_commune TEXT,
    code_postal BIGINT,
    code_insee BIGINT
);