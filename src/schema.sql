
CREATE TABLE images (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    path   TEXT NOT NULL UNIQUE,
    label  TEXT NOT NULL CHECK (label IN ('circle','triangle','square','pentagon','hexagon')),
    source TEXT NOT NULL,
    split  TEXT NOT NULL CHECK (split IN ('train','test'))
);
