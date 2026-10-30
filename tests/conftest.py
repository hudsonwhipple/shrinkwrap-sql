import sqlite3

import pytest


@pytest.fixture
def tiny_db(tmp_path):
    path = tmp_path / "tiny.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER, country TEXT);
        CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, singer_id INTEGER REFERENCES singer, year INTEGER);
        INSERT INTO singer VALUES (1, 'Ana', 31, 'France'), (2, 'Bo', 25, 'Japan'), (3, 'Cy', 40, 'France');
        INSERT INTO concert VALUES (1, 1, 2014), (2, 1, 2015), (3, 3, 2015);
        """
    )
    conn.commit()
    conn.close()
    return path
